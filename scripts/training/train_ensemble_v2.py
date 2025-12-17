"""
改善版アンサンブル学習スクリプト v2
- 複数アーキテクチャによる多様性向上
- クラス不均衡対応（重み付け損失）
- 重み付き投票
- バギング（ブートストラップサンプリング）
"""
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
from collections import Counter

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATASET_DIR = PROJECT_ROOT / "data/datasets/dataset"
ENSEMBLE_DIR = PROJECT_ROOT / "models/ensemble_models_v2"

NUM_POINTS = 1024
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

# デバイス設定
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


class FurnitureDataset(Dataset):
    def __init__(self, root_dir, num_points=1024, augment=False):
        self.num_points = num_points
        self.augment = augment
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
        # Y軸回転
        theta = np.random.uniform(0, 2 * np.pi)
        Ry = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ], dtype=np.float32)
        points = points @ Ry.T

        # ミラーリング
        if np.random.random() > 0.5:
            points[:, 0] = -points[:, 0]

        # スケール
        scale = np.random.uniform(0.9, 1.1, size=3).astype(np.float32)
        points *= scale

        # ノイズ
        points += np.random.normal(0, 0.02, points.shape).astype(np.float32)

        return points


# =============================================================================
# 多様なアーキテクチャ
# =============================================================================

class PointNetEncoder(nn.Module):
    """標準PointNet"""
    def __init__(self, feature_dim=512):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, feature_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(feature_dim)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = torch.max(x, dim=2)[0]
        return x


class PointNetEncoderDeep(nn.Module):
    """より深いPointNet"""
    def __init__(self, feature_dim=512):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.conv4 = nn.Conv1d(128, 256, 1)
        self.conv5 = nn.Conv1d(256, 512, 1)
        self.conv6 = nn.Conv1d(512, feature_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(512)
        self.bn6 = nn.BatchNorm1d(feature_dim)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = torch.relu(self.bn5(self.conv5(x)))
        x = torch.relu(self.bn6(self.conv6(x)))
        x = torch.max(x, dim=2)[0]
        return x


class PointNetEncoderWide(nn.Module):
    """より広いPointNet"""
    def __init__(self, feature_dim=512):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.conv3 = nn.Conv1d(256, 512, 1)
        self.conv4 = nn.Conv1d(512, feature_dim, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(feature_dim)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = torch.max(x, dim=2)[0]
        return x


class PointNetWithLocalFeatures(nn.Module):
    """ローカル＋グローバル特徴を結合"""
    def __init__(self, feature_dim=512):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)

        # ローカル特徴とグローバル特徴を結合後の処理
        self.conv4 = nn.Conv1d(256 + 256, feature_dim, 1)
        self.bn4 = nn.BatchNorm1d(feature_dim)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        local_feat = torch.relu(self.bn3(self.conv3(x)))  # [B, 256, N]

        global_feat = torch.max(local_feat, dim=2, keepdim=True)[0]  # [B, 256, 1]
        global_feat = global_feat.expand(-1, -1, local_feat.size(2))  # [B, 256, N]

        combined = torch.cat([local_feat, global_feat], dim=1)  # [B, 512, N]
        x = torch.relu(self.bn4(self.conv4(combined)))
        x = torch.max(x, dim=2)[0]
        return x


class DGCNNEncoder(nn.Module):
    """DGCNN風のエンコーダー（簡易版EdgeConv）"""
    def __init__(self, feature_dim=512, k=20):
        super().__init__()
        self.k = k

        self.conv1 = nn.Conv2d(6, 64, 1)
        self.conv2 = nn.Conv2d(128, 128, 1)
        self.conv3 = nn.Conv2d(256, 256, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv_out = nn.Conv1d(64 + 128 + 256, feature_dim, 1)
        self.bn_out = nn.BatchNorm1d(feature_dim)

    def knn(self, x):
        # x: [B, 3, N]
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        idx = pairwise_distance.topk(k=self.k, dim=-1)[1]
        return idx

    def get_graph_feature(self, x, idx=None):
        batch_size, num_dims, num_points = x.size()
        if idx is None:
            idx = self.knn(x)

        idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.reshape(-1)

        x = x.transpose(2, 1).contiguous()
        feature = x.reshape(batch_size * num_points, -1)[idx, :]
        feature = feature.reshape(batch_size, num_points, self.k, num_dims)
        x = x.reshape(batch_size, num_points, 1, num_dims).repeat(1, 1, self.k, 1)
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
        return feature

    def forward(self, x):
        batch_size = x.size(0)

        x1 = self.get_graph_feature(x)
        x1 = torch.relu(self.bn1(self.conv1(x1)))
        x1 = x1.max(dim=-1)[0]

        x2 = self.get_graph_feature(x1)
        x2 = torch.relu(self.bn2(self.conv2(x2)))
        x2 = x2.max(dim=-1)[0]

        x3 = self.get_graph_feature(x2)
        x3 = torch.relu(self.bn3(self.conv3(x3)))
        x3 = x3.max(dim=-1)[0]

        x = torch.cat([x1, x2, x3], dim=1)
        x = torch.relu(self.bn_out(self.conv_out(x)))
        x = torch.max(x, dim=2)[0]
        return x


class StabilityClassifier(nn.Module):
    def __init__(self, encoder_type='standard', feature_dim=512, dropout=0.5):
        super().__init__()

        if encoder_type == 'standard':
            self.encoder = PointNetEncoder(feature_dim)
        elif encoder_type == 'deep':
            self.encoder = PointNetEncoderDeep(feature_dim)
        elif encoder_type == 'wide':
            self.encoder = PointNetEncoderWide(feature_dim)
        elif encoder_type == 'local':
            self.encoder = PointNetWithLocalFeatures(feature_dim)
        elif encoder_type == 'dgcnn':
            self.encoder = DGCNNEncoder(feature_dim)
        else:
            self.encoder = PointNetEncoder(feature_dim)

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
# モデル設定（多様性確保）
# =============================================================================

MODEL_CONFIGS = [
    {'encoder_type': 'standard', 'feature_dim': 512, 'dropout': 0.5, 'lr': 0.001, 'seed': 42},
    {'encoder_type': 'deep', 'feature_dim': 512, 'dropout': 0.4, 'lr': 0.0008, 'seed': 123},
    {'encoder_type': 'wide', 'feature_dim': 512, 'dropout': 0.5, 'lr': 0.001, 'seed': 456},
    {'encoder_type': 'local', 'feature_dim': 512, 'dropout': 0.4, 'lr': 0.0008, 'seed': 789},
    {'encoder_type': 'dgcnn', 'feature_dim': 512, 'dropout': 0.5, 'lr': 0.001, 'seed': 1024},
    {'encoder_type': 'standard', 'feature_dim': 768, 'dropout': 0.6, 'lr': 0.0005, 'seed': 2048},
    {'encoder_type': 'deep', 'feature_dim': 768, 'dropout': 0.5, 'lr': 0.0008, 'seed': 4096},
]


def compute_class_weights(labels):
    """クラス不均衡に対応する重みを計算"""
    counter = Counter(labels)
    total = len(labels)
    weights = {cls: total / count for cls, count in counter.items()}
    return weights


def train_single_model(train_loader, val_loader, config, model_id, class_weights):
    """1つのモデルを訓練"""
    set_seed(config['seed'])

    model = StabilityClassifier(
        encoder_type=config['encoder_type'],
        feature_dim=config['feature_dim'],
        dropout=config['dropout']
    ).to(DEVICE)

    # クラス重みを適用した損失関数
    weight_tensor = torch.tensor([class_weights[0], class_weights[1]], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.1)

    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0
    best_state = None
    patience = 30
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        # 訓練
        model.train()
        train_loss = 0
        for points, labels in train_loader:
            points, labels = points.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(points)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # 検証
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for points, labels in val_loader:
                points, labels = points.to(DEVICE), labels.to(DEVICE)
                outputs = model(points)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_acc = 100. * correct / total
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 20 == 0:
            print(f"  [{config['encoder_type']}] Epoch {epoch}: Val Acc = {val_acc:.2f}% (Best: {best_val_acc:.2f}%)")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    # ベストモデルを保存
    model_path = ENSEMBLE_DIR / f"model_{model_id}.pth"
    torch.save({
        'model_state_dict': best_state,
        'val_acc': best_val_acc,
        'config': config
    }, model_path)

    return best_val_acc


def evaluate_ensemble_weighted(models, configs, val_loader):
    """重み付きアンサンブル評価"""
    # 検証精度に基づく重み
    val_accs = [cfg.get('val_acc', 90.0) for cfg in configs]
    weights = np.array(val_accs) ** 2  # 精度の2乗で重み付け
    weights = weights / weights.sum()

    correct = 0
    total = 0

    for model in models:
        model.eval()

    with torch.no_grad():
        for points, labels in val_loader:
            points, labels = points.to(DEVICE), labels.to(DEVICE)

            # 各モデルの予測を収集
            weighted_probs = None
            for i, model in enumerate(models):
                outputs = model(points)
                probs = torch.softmax(outputs, dim=1)
                if weighted_probs is None:
                    weighted_probs = weights[i] * probs
                else:
                    weighted_probs += weights[i] * probs

            _, predicted = weighted_probs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100. * correct / total, weights


def main():
    print("\n" + "=" * 70)
    print("改善版アンサンブル学習 v2")
    print("- 複数アーキテクチャ（PointNet, Deep, Wide, Local, DGCNN）")
    print("- クラス不均衡対応")
    print("- 重み付き投票")
    print("=" * 70)

    # ディレクトリ作成
    ENSEMBLE_DIR.mkdir(exist_ok=True)

    # データセット読み込み
    full_dataset = FurnitureDataset(DATASET_DIR, num_points=NUM_POINTS, augment=True)
    labels = full_dataset.get_labels()

    print(f"\nデータセット: {len(full_dataset)} サンプル")
    print(f"  Stable: {labels.count(1)}, Unstable: {labels.count(0)}")

    # クラス重み計算
    class_weights = compute_class_weights(labels)
    print(f"クラス重み: {class_weights}")

    # 学習/検証分割（層化分割）
    set_seed(42)
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)

    train_size = int(0.8 * len(full_dataset))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # 検証データセット（拡張なし）
    val_dataset = Subset(full_dataset, val_indices)
    val_dataset.dataset.augment = False
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"学習データ: {len(train_indices)}, 検証データ: {len(val_indices)}")
    print(f"デバイス: {DEVICE}")
    print(f"モデル数: {len(MODEL_CONFIGS)}")

    # 各モデルを訓練
    individual_accs = []
    trained_configs = []

    for i, config in enumerate(MODEL_CONFIGS):
        print(f"\n{'─' * 70}")
        print(f"モデル {i+1}/{len(MODEL_CONFIGS)}: {config['encoder_type']} (dim={config['feature_dim']}, dropout={config['dropout']})")
        print(f"{'─' * 70}")

        # バギング：各モデルで異なるサブセットを使用
        set_seed(config['seed'])
        bootstrap_indices = np.random.choice(train_indices, len(train_indices), replace=True).tolist()

        train_dataset = Subset(full_dataset, bootstrap_indices)
        train_dataset.dataset.augment = True

        # シャッフルでデータローダー作成（クラス重みは損失関数で対応）
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        acc = train_single_model(train_loader, val_loader, config, i, class_weights)
        individual_accs.append(acc)

        config_with_acc = config.copy()
        config_with_acc['val_acc'] = acc
        trained_configs.append(config_with_acc)

        print(f"  ベスト精度: {acc:.2f}%")

    # アンサンブル評価
    print(f"\n{'=' * 70}")
    print("アンサンブル評価")
    print(f"{'=' * 70}")

    models = []
    for i in range(len(MODEL_CONFIGS)):
        config = MODEL_CONFIGS[i]
        model = StabilityClassifier(
            encoder_type=config['encoder_type'],
            feature_dim=config['feature_dim'],
            dropout=config['dropout']
        ).to(DEVICE)
        checkpoint = torch.load(ENSEMBLE_DIR / f"model_{i}.pth", map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        models.append(model)

    ensemble_acc, weights = evaluate_ensemble_weighted(models, trained_configs, val_loader)

    print("\n結果:")
    print("-" * 50)
    for i, (acc, config) in enumerate(zip(individual_accs, MODEL_CONFIGS)):
        print(f"  モデル {i+1} [{config['encoder_type']:8s}]: {acc:.2f}% (重み: {weights[i]:.3f})")
    print("-" * 50)
    print(f"  個別モデル平均:     {np.mean(individual_accs):.2f}%")
    print(f"  重み付きアンサンブル: {ensemble_acc:.2f}%")
    print(f"  改善: {ensemble_acc - np.mean(individual_accs):+.2f}%")
    print("=" * 70)

    # 結果保存
    results = {
        "individual_accuracies": individual_accs,
        "mean_accuracy": float(np.mean(individual_accs)),
        "ensemble_accuracy": ensemble_acc,
        "num_models": len(MODEL_CONFIGS),
        "dataset_size": len(full_dataset),
        "model_configs": [
            {
                'encoder_type': c['encoder_type'],
                'feature_dim': c['feature_dim'],
                'dropout': c['dropout'],
                'val_acc': c.get('val_acc', 0),
                'weight': float(weights[i])
            }
            for i, c in enumerate(trained_configs)
        ]
    }
    with open(ENSEMBLE_DIR / "results.json", 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nモデル保存先: {ENSEMBLE_DIR}")


if __name__ == "__main__":
    main()
