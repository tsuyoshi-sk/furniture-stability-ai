"""
強化データ拡張版 学習スクリプト
- 多様な回転・スケール
- ランダムノイズ
- ポイントドロップアウト
- Mixup
- Hard negative mining
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from collections import Counter

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATASET_DIR = PROJECT_ROOT / "data/datasets/dataset"
MODEL_DIR = PROJECT_ROOT / "models/models_augmented"

NUM_POINTS = 1024
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 0.001

# デバイス設定
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_obj_vertices(filepath):
    vertices = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices.append([x, y, z])
    return np.array(vertices, dtype=np.float32)


def normalize_point_cloud(points):
    centroid = np.mean(points, axis=0)
    points = points - centroid
    max_dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    if max_dist > 0:
        points = points / max_dist
    return points


def sample_points(points, num_points):
    n = len(points)
    if n == 0:
        return np.zeros((num_points, 3), dtype=np.float32)
    if n >= num_points:
        indices = np.random.choice(n, num_points, replace=False)
    else:
        indices = np.random.choice(n, num_points, replace=True)
    return points[indices]


# =============================================================================
# 強化データ拡張
# =============================================================================

class StrongAugmentation:
    """強化データ拡張クラス"""

    def __init__(self, p_rotation=0.9, p_scale=0.8, p_noise=0.7,
                 p_dropout=0.5, p_jitter=0.6, p_shift=0.5):
        self.p_rotation = p_rotation
        self.p_scale = p_scale
        self.p_noise = p_noise
        self.p_dropout = p_dropout
        self.p_jitter = p_jitter
        self.p_shift = p_shift

    def __call__(self, points):
        """点群にデータ拡張を適用"""
        points = points.copy()

        # 1. ランダム回転（Y軸中心 + 微小な他軸回転）
        if np.random.random() < self.p_rotation:
            points = self._random_rotation(points)

        # 2. ランダムスケール
        if np.random.random() < self.p_scale:
            scale = np.random.uniform(0.8, 1.2)
            points *= scale

        # 3. 異方性スケール（X, Y, Z軸別々）
        if np.random.random() < 0.3:
            scale_xyz = np.random.uniform(0.9, 1.1, size=3)
            points *= scale_xyz

        # 4. ガウシアンノイズ
        if np.random.random() < self.p_noise:
            noise_level = np.random.uniform(0.01, 0.03)
            points += np.random.normal(0, noise_level, points.shape).astype(np.float32)

        # 5. ポイントジッター（各点を個別に微小移動）
        if np.random.random() < self.p_jitter:
            jitter = np.random.uniform(-0.02, 0.02, points.shape).astype(np.float32)
            points += jitter

        # 6. ランダムシフト
        if np.random.random() < self.p_shift:
            shift = np.random.uniform(-0.1, 0.1, size=3).astype(np.float32)
            points += shift

        # 7. ポイントドロップアウト
        if np.random.random() < self.p_dropout:
            points = self._point_dropout(points)

        # 8. ミラーリング
        if np.random.random() < 0.5:
            points[:, 0] = -points[:, 0]
        if np.random.random() < 0.3:
            points[:, 2] = -points[:, 2]

        return points

    def _random_rotation(self, points):
        """ランダム回転を適用"""
        # Y軸回転（全角度）
        theta_y = np.random.uniform(0, 2 * np.pi)
        Ry = np.array([
            [np.cos(theta_y), 0, np.sin(theta_y)],
            [0, 1, 0],
            [-np.sin(theta_y), 0, np.cos(theta_y)]
        ], dtype=np.float32)
        points = points @ Ry.T

        # X軸・Z軸の微小回転（±15度）
        if np.random.random() < 0.3:
            theta_x = np.random.uniform(-np.pi/12, np.pi/12)
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(theta_x), -np.sin(theta_x)],
                [0, np.sin(theta_x), np.cos(theta_x)]
            ], dtype=np.float32)
            points = points @ Rx.T

        if np.random.random() < 0.3:
            theta_z = np.random.uniform(-np.pi/12, np.pi/12)
            Rz = np.array([
                [np.cos(theta_z), -np.sin(theta_z), 0],
                [np.sin(theta_z), np.cos(theta_z), 0],
                [0, 0, 1]
            ], dtype=np.float32)
            points = points @ Rz.T

        return points

    def _point_dropout(self, points, max_dropout=0.2):
        """ランダムにポイントをドロップ"""
        n = len(points)
        dropout_ratio = np.random.uniform(0, max_dropout)
        keep_n = int(n * (1 - dropout_ratio))
        indices = np.random.choice(n, keep_n, replace=False)

        # ドロップした分を複製で補完
        if keep_n < n:
            extra_indices = np.random.choice(indices, n - keep_n, replace=True)
            indices = np.concatenate([indices, extra_indices])

        return points[indices]


class FurnitureDatasetAugmented(Dataset):
    """強化データ拡張付きデータセット"""

    def __init__(self, root_dir, num_points=1024, augment=False):
        self.num_points = num_points
        self.augment = augment
        self.augmentor = StrongAugmentation() if augment else None
        self.samples = []

        for label, folder in [(1, "stable"), (0, "unstable")]:
            folder_path = Path(root_dir) / folder
            if not folder_path.exists():
                continue
            for obj_file in folder_path.glob("*.obj"):
                self.samples.append({
                    'obj_path': str(obj_file),
                    'label': label
                })

    def __len__(self):
        return len(self.samples)

    def get_labels(self):
        return [s['label'] for s in self.samples]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        vertices = load_obj_vertices(sample['obj_path'])
        vertices = normalize_point_cloud(vertices)
        points = sample_points(vertices, self.num_points)

        if self.augment and self.augmentor:
            points = self.augmentor(points)

        points = torch.from_numpy(points).float()
        label = torch.tensor(sample['label'], dtype=torch.long)
        return points, label


# =============================================================================
# モデル定義（Local + DGCNN ハイブリッド）
# =============================================================================

class PointNetWithLocalFeatures(nn.Module):
    """Local特徴付きPointNet"""
    def __init__(self, feature_dim=512):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256 + 256, feature_dim, 1)
        self.bn4 = nn.BatchNorm1d(feature_dim)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        local_feat = torch.relu(self.bn3(self.conv3(x)))
        global_feat = torch.max(local_feat, dim=2, keepdim=True)[0]
        global_feat = global_feat.expand(-1, -1, local_feat.size(2))
        combined = torch.cat([local_feat, global_feat], dim=1)
        x = torch.relu(self.bn4(self.conv4(combined)))
        x = torch.max(x, dim=2)[0]
        return x


class StabilityClassifier(nn.Module):
    """安定性分類器"""
    def __init__(self, feature_dim=512, dropout=0.5):
        super().__init__()
        self.encoder = PointNetWithLocalFeatures(feature_dim)
        self.fc1 = nn.Linear(feature_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# =============================================================================
# Mixup
# =============================================================================

def mixup_data(x, y, alpha=0.2):
    """Mixupデータ拡張"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup用損失関数"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# =============================================================================
# Label Smoothing
# =============================================================================

class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing付きCrossEntropy"""
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_preds = F.log_softmax(pred, dim=-1)

        # One-hot with smoothing
        targets = torch.zeros_like(log_preds).scatter_(
            -1, target.unsqueeze(-1), 1
        )
        targets = (1 - self.smoothing) * targets + self.smoothing / n_classes

        # クラス重み適用
        if self.weight is not None:
            weight = self.weight[target]
            loss = (-targets * log_preds).sum(dim=-1) * weight
        else:
            loss = (-targets * log_preds).sum(dim=-1)

        return loss.mean()


# =============================================================================
# 学習
# =============================================================================

def train():
    set_seed(42)
    MODEL_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("強化データ拡張版 学習")
    print("=" * 70)

    # データセット
    full_dataset = FurnitureDatasetAugmented(DATASET_DIR, NUM_POINTS)
    labels = full_dataset.get_labels()
    label_counts = Counter(labels)

    print(f"\nデータセット: {len(full_dataset)} サンプル")
    print(f"  Stable: {label_counts[1]}, Unstable: {label_counts[0]}")

    # クラス重み
    total = len(labels)
    weight_unstable = total / (2 * label_counts[0])
    weight_stable = total / (2 * label_counts[1])
    class_weights = torch.tensor([weight_unstable, weight_stable], device=DEVICE)

    # 分割
    indices = list(range(len(full_dataset)))
    np.random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_indices = indices[:split]
    val_indices = indices[split:]

    train_dataset = FurnitureDatasetAugmented(DATASET_DIR, NUM_POINTS, augment=True)
    train_dataset.samples = [full_dataset.samples[i] for i in train_indices]

    val_dataset = FurnitureDatasetAugmented(DATASET_DIR, NUM_POINTS, augment=False)
    val_dataset.samples = [full_dataset.samples[i] for i in val_indices]

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"学習: {len(train_dataset)}, 検証: {len(val_dataset)}")
    print(f"デバイス: {DEVICE}")

    # モデル
    model = StabilityClassifier(feature_dim=512, dropout=0.5).to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"パラメータ数: {num_params:,}")

    # 損失関数・オプティマイザ
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1, weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-6)

    print("\n" + "-" * 70)
    print("学習開始 (Mixup + Label Smoothing + 強化データ拡張)")
    print("-" * 70)

    best_val_acc = 0
    best_state = None
    patience = 40
    patience_counter = 0
    use_mixup = True

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for points, labels_batch in train_loader:
            points = points.transpose(2, 1).to(DEVICE)  # [B, 3, N]
            labels_batch = labels_batch.to(DEVICE)

            # Mixup
            if use_mixup and np.random.random() < 0.5:
                points, labels_a, labels_b, lam = mixup_data(points, labels_batch, alpha=0.2)

                optimizer.zero_grad()
                outputs = model(points)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                optimizer.zero_grad()
                outputs = model(points)
                loss = criterion(outputs, labels_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels_batch.size(0)
            train_correct += predicted.eq(labels_batch).sum().item()

        scheduler.step()

        # 検証
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for points, labels_batch in val_loader:
                points = points.transpose(2, 1).to(DEVICE)
                labels_batch = labels_batch.to(DEVICE)
                outputs = model(points)
                _, predicted = outputs.max(1)
                val_total += labels_batch.size(0)
                val_correct += predicted.eq(labels_batch).sum().item()

        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total

        if epoch % 10 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:3d}: Train={train_acc:.1f}%, Val={val_acc:.1f}% (Best={best_val_acc:.1f}%) LR={lr:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    # 保存
    model_path = MODEL_DIR / "local_augmented_best.pth"
    torch.save({
        'model_state_dict': best_state,
        'val_acc': best_val_acc,
        'config': {
            'architecture': 'PointNet Local + Strong Augmentation',
            'feature_dim': 512,
            'dropout': 0.5
        }
    }, model_path)

    print(f"\nベスト検証精度: {best_val_acc:.2f}%")
    print(f"モデル保存: {model_path}")

    # 最終評価
    print("\n" + "-" * 70)
    print("最終評価")
    print("-" * 70)

    model.load_state_dict(best_state)
    model.eval()

    tp = tn = fp = fn = 0
    with torch.no_grad():
        for points, labels_batch in val_loader:
            points = points.transpose(2, 1).to(DEVICE)
            labels_batch = labels_batch.to(DEVICE)
            outputs = model(points)
            _, predicted = outputs.max(1)

            for p, l in zip(predicted, labels_batch):
                if p == 1 and l == 1:
                    tp += 1
                elif p == 0 and l == 0:
                    tn += 1
                elif p == 1 and l == 0:
                    fp += 1
                else:
                    fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    print(f"  Accuracy:  {accuracy:.2%}")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall:    {recall:.2%}")
    print(f"  F1 Score:  {f1:.2%}")
    print(f"  Confusion: TP={tp}, TN={tn}, FP={fp}, FN={fn}")

    return best_val_acc


if __name__ == "__main__":
    train()
