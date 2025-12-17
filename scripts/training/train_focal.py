"""
Focal Loss + 強化データ拡張による学習スクリプト
- Focal Lossで難しいサンプルに重点
- より強いデータ拡張
- Localモデル（最高性能）を使用
"""
import os
import json
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
MODEL_DIR = PROJECT_ROOT / "models/models_focal"

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
# Focal Loss
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss: 難しいサンプルに重点を置く損失関数

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # クラス重み [unstable, stable]
        self.gamma = gamma  # フォーカシングパラメータ（大きいほど難サンプル重視）
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # 正解クラスの確率

        # alpha重み
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * ((1 - pt) ** self.gamma) * ce_loss
        else:
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# =============================================================================
# 強化データ拡張
# =============================================================================

class FurnitureDataset(Dataset):
    def __init__(self, root_dir, num_points=1024, augment=False, strong_augment=False):
        self.num_points = num_points
        self.augment = augment
        self.strong_augment = strong_augment
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

        if self.augment:
            points = self._augment(points)

        points = torch.from_numpy(points).float().transpose(0, 1)
        label = torch.tensor(sample['label'], dtype=torch.long)
        return points, label

    def _augment(self, points):
        # Y軸回転（ランダム角度）
        theta = np.random.uniform(0, 2 * np.pi)
        Ry = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ], dtype=np.float32)
        points = points @ Ry.T

        # X軸ミラーリング
        if np.random.random() > 0.5:
            points[:, 0] = -points[:, 0]

        # Z軸ミラーリング
        if np.random.random() > 0.5:
            points[:, 2] = -points[:, 2]

        # スケール変動
        scale = np.random.uniform(0.85, 1.15, size=3).astype(np.float32)
        points *= scale

        # ジッター（ノイズ）
        points += np.random.normal(0, 0.02, points.shape).astype(np.float32)

        if self.strong_augment:
            # X軸回転（微小）
            angle_x = np.random.uniform(-0.1, 0.1)
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(angle_x), -np.sin(angle_x)],
                [0, np.sin(angle_x), np.cos(angle_x)]
            ], dtype=np.float32)
            points = points @ Rx.T

            # Z軸回転（微小）
            angle_z = np.random.uniform(-0.1, 0.1)
            Rz = np.array([
                [np.cos(angle_z), -np.sin(angle_z), 0],
                [np.sin(angle_z), np.cos(angle_z), 0],
                [0, 0, 1]
            ], dtype=np.float32)
            points = points @ Rz.T

            # ランダムドロップアウト（点の一部を削除して再サンプル）
            if np.random.random() > 0.7:
                keep_ratio = np.random.uniform(0.8, 0.95)
                n_keep = int(len(points) * keep_ratio)
                indices = np.random.choice(len(points), n_keep, replace=False)
                points_kept = points[indices]
                # 足りない分を複製
                n_need = self.num_points - n_keep
                extra_indices = np.random.choice(n_keep, n_need, replace=True)
                points = np.vstack([points_kept, points_kept[extra_indices]])

        return points


# =============================================================================
# PointNet with Local Features（最高性能モデル）
# =============================================================================

class PointNetWithLocalFeatures(nn.Module):
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
    def __init__(self, feature_dim=512, dropout=0.4):
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
# 学習
# =============================================================================

def train():
    set_seed(42)
    MODEL_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("Focal Loss + 強化データ拡張 学習")
    print("=" * 70)

    # データセット
    full_dataset = FurnitureDataset(DATASET_DIR, NUM_POINTS)
    labels = full_dataset.get_labels()
    label_counts = Counter(labels)

    print(f"\nデータセット: {len(full_dataset)} サンプル")
    print(f"  Stable: {label_counts[1]}, Unstable: {label_counts[0]}")

    # クラス重み計算
    total = len(labels)
    weight_unstable = total / (2 * label_counts[0])
    weight_stable = total / (2 * label_counts[1])
    alpha = torch.tensor([weight_unstable, weight_stable], device=DEVICE)
    print(f"クラス重み: unstable={weight_unstable:.3f}, stable={weight_stable:.3f}")

    # 学習/検証分割
    indices = list(range(len(full_dataset)))
    np.random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_indices = indices[:split]
    val_indices = indices[split:]

    # 学習データセット（強化拡張あり）
    train_dataset = FurnitureDataset(DATASET_DIR, NUM_POINTS, augment=True, strong_augment=True)
    train_dataset.samples = [full_dataset.samples[i] for i in train_indices]

    # 検証データセット
    val_dataset = FurnitureDataset(DATASET_DIR, NUM_POINTS, augment=False)
    val_dataset.samples = [full_dataset.samples[i] for i in val_indices]

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"学習: {len(train_dataset)}, 検証: {len(val_dataset)}")
    print(f"デバイス: {DEVICE}")

    # モデル
    model = StabilityClassifier(feature_dim=512, dropout=0.4).to(DEVICE)

    # Focal Loss（gamma=1.0で緩やかなフォーカス）
    criterion = FocalLoss(alpha=alpha, gamma=1.0)

    # オプティマイザ（AdamW + weight decay）
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # 学習率スケジューラ（コサインアニーリング）
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    print("\n" + "-" * 70)
    print("学習開始")
    print("-" * 70)

    best_val_acc = 0
    best_state = None
    patience = 50
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        # 学習
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for points, labels in train_loader:
            points, labels = points.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(points)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        scheduler.step()

        # 検証
        model.eval()
        val_correct = 0
        val_total = 0
        val_tp = val_tn = val_fp = val_fn = 0

        with torch.no_grad():
            for points, labels in val_loader:
                points, labels = points.to(DEVICE), labels.to(DEVICE)
                outputs = model(points)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                # 混同行列
                for p, l in zip(predicted, labels):
                    if p == 1 and l == 1:
                        val_tp += 1
                    elif p == 0 and l == 0:
                        val_tn += 1
                    elif p == 1 and l == 0:
                        val_fp += 1
                    else:
                        val_fn += 1

        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total

        # 20エポックごとに表示
        if epoch % 20 == 0 or epoch == 1:
            precision = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0
            recall = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            print(f"  Epoch {epoch:3d}: Train={train_acc:.1f}%, Val={val_acc:.1f}% "
                  f"(Best={best_val_acc:.1f}%) P={precision:.2f} R={recall:.2f} F1={f1:.2f}")

        # ベストモデル保存
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    # モデル保存
    model_path = MODEL_DIR / "model_focal.pth"
    torch.save({
        'model_state_dict': best_state,
        'val_acc': best_val_acc,
        'config': {
            'encoder_type': 'local',
            'feature_dim': 512,
            'dropout': 0.4
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

    val_tp = val_tn = val_fp = val_fn = 0
    with torch.no_grad():
        for points, labels in val_loader:
            points, labels = points.to(DEVICE), labels.to(DEVICE)
            outputs = model(points)
            _, predicted = outputs.max(1)

            for p, l in zip(predicted, labels):
                if p == 1 and l == 1:
                    val_tp += 1
                elif p == 0 and l == 0:
                    val_tn += 1
                elif p == 1 and l == 0:
                    val_fp += 1
                else:
                    val_fn += 1

    precision = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0
    recall = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (val_tp + val_tn) / (val_tp + val_tn + val_fp + val_fn)

    print(f"  Accuracy:  {accuracy:.2%}")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall:    {recall:.2%}")
    print(f"  F1 Score:  {f1:.2%}")
    print(f"  Confusion: TP={val_tp}, TN={val_tn}, FP={val_fp}, FN={val_fn}")

    return best_val_acc


if __name__ == "__main__":
    train()
