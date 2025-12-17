#!/usr/bin/env python3
"""
統合データセット（Okamura + Original）での安定性モデル学習
- dataset_unified/からlabels.jsonを読み込み
- 11,600+モデルで学習
- アンサンブル学習
"""
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATASET_DIR = PROJECT_ROOT / "data/datasets/dataset_unified"
MODEL_DIR = PROJECT_ROOT / "models/models_unified"

NUM_POINTS = 1024
BATCH_SIZE = 32
EPOCHS = 50  # 大規模データなので少なめ
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


def load_obj_vertices(filepath):
    vertices = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    vertices.append([x, y, z])
    except:
        pass
    return np.array(vertices, dtype=np.float32) if vertices else np.zeros((0, 3), dtype=np.float32)


def normalize_point_cloud(points):
    if len(points) == 0:
        return points
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


class UnifiedDataset(Dataset):
    """統合データセット"""
    def __init__(self, root_dir, num_points=1024, augment=False, indices=None):
        self.root_dir = Path(root_dir)
        self.num_points = num_points
        self.augment = augment

        # labels.jsonを読み込み
        with open(self.root_dir / 'labels.json', 'r') as f:
            data = json.load(f)

        self.samples = []
        for item in data['items']:
            obj_path = self.root_dir / item['filename']
            if obj_path.exists():
                self.samples.append({
                    'obj_path': str(obj_path),
                    'label': item.get('stable', 1),
                    'shape': item.get('shape', 'unknown'),
                    'source': item.get('source', 'unknown')
                })

        # インデックスでフィルタ
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

    def __len__(self):
        return len(self.samples)

    def get_labels(self):
        return [s['label'] for s in self.samples]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        vertices = load_obj_vertices(sample['obj_path'])

        if len(vertices) == 0:
            # 空のファイルの場合はダミーデータ
            points = np.zeros((self.num_points, 3), dtype=np.float32)
        else:
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
        points += np.random.normal(0, 0.01, points.shape).astype(np.float32)

        return points


# =============================================================================
# モデル定義
# =============================================================================

class PointNetEncoder(nn.Module):
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


class StabilityClassifier(nn.Module):
    def __init__(self, encoder_type='standard', feature_dim=512, dropout=0.5):
        super().__init__()

        if encoder_type == 'deep':
            self.encoder = PointNetEncoderDeep(feature_dim)
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


def compute_class_weights(labels):
    counter = Counter(labels)
    total = len(labels)
    weights = {cls: total / (len(counter) * count) for cls, count in counter.items()}
    return weights


def train_model(train_loader, val_loader, config, class_weights, model_name):
    """モデル訓練"""
    set_seed(config.get('seed', 42))

    model = StabilityClassifier(
        encoder_type=config['encoder_type'],
        feature_dim=config['feature_dim'],
        dropout=config['dropout']
    ).to(DEVICE)

    # クラス重み
    weight_tensor = torch.tensor([class_weights.get(0, 1.0), class_weights.get(1, 1.0)],
                                  dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.1)

    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0
    best_state = None
    patience = 15
    patience_counter = 0

    print(f"\n訓練開始: {model_name}")

    for epoch in range(1, EPOCHS + 1):
        # 訓練
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = 100. * train_correct / train_total

        # 検証
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for points, labels in val_loader:
                points, labels = points.to(DEVICE), labels.to(DEVICE)
                outputs = model(points)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: Train={train_acc:.2f}%, Val={val_acc:.2f}% (Best: {best_val_acc:.2f}%)")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    return model, best_state, best_val_acc


def main():
    print("\n" + "=" * 70)
    print("統合データセットでの安定性モデル学習")
    print("=" * 70)

    MODEL_DIR.mkdir(exist_ok=True)

    # データ確認
    with open(DATASET_DIR / 'labels.json', 'r') as f:
        data = json.load(f)

    print(f"\nデータセット: {DATASET_DIR}")
    print(f"総モデル数: {data['total_models']}")
    print(f"ソース別:")
    for src, count in data.get('sources', {}).items():
        print(f"  {src}: {count}")
    print(f"安定: {data['stability']['stable']}, 不安定: {data['stability']['unstable']}")

    # データセット読み込み
    print(f"\nデータ読み込み中...")
    full_dataset = UnifiedDataset(DATASET_DIR, num_points=NUM_POINTS, augment=False)
    labels = full_dataset.get_labels()

    print(f"有効サンプル数: {len(full_dataset)}")

    # クラス重み
    class_weights = compute_class_weights(labels)
    print(f"クラス重み: {class_weights}")

    # Train/Val分割
    indices = list(range(len(full_dataset)))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=42,
        stratify=labels
    )

    print(f"学習データ: {len(train_idx)}, 検証データ: {len(val_idx)}")
    print(f"デバイス: {DEVICE}")

    # データセット作成
    train_dataset = UnifiedDataset(DATASET_DIR, num_points=NUM_POINTS, augment=True, indices=train_idx)
    val_dataset = UnifiedDataset(DATASET_DIR, num_points=NUM_POINTS, augment=False, indices=val_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # モデル設定
    model_configs = [
        {'encoder_type': 'standard', 'feature_dim': 512, 'dropout': 0.5, 'lr': 0.001, 'seed': 42},
        {'encoder_type': 'deep', 'feature_dim': 512, 'dropout': 0.4, 'lr': 0.0008, 'seed': 123},
        {'encoder_type': 'standard', 'feature_dim': 768, 'dropout': 0.5, 'lr': 0.0005, 'seed': 456},
    ]

    # 各モデルを訓練
    results = []

    for i, config in enumerate(model_configs):
        model_name = f"model_{i}_{config['encoder_type']}"
        model, best_state, val_acc = train_model(
            train_loader, val_loader, config, class_weights, model_name
        )

        # 保存
        save_path = MODEL_DIR / f"{model_name}.pth"
        torch.save({
            'model_state_dict': best_state,
            'config': config,
            'val_acc': val_acc
        }, save_path)

        results.append({
            'name': model_name,
            'val_acc': val_acc,
            'config': config
        })

        print(f"  保存: {save_path} (Val: {val_acc:.2f}%)")

    # アンサンブル評価
    print(f"\n{'=' * 70}")
    print("アンサンブル評価")
    print(f"{'=' * 70}")

    models = []
    for r in results:
        config = r['config']
        model = StabilityClassifier(
            encoder_type=config['encoder_type'],
            feature_dim=config['feature_dim'],
            dropout=config['dropout']
        ).to(DEVICE)
        checkpoint = torch.load(MODEL_DIR / f"{r['name']}.pth", map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models.append(model)

    # 重み付きアンサンブル
    weights = np.array([r['val_acc'] for r in results]) ** 2
    weights = weights / weights.sum()

    correct = 0
    total = 0

    with torch.no_grad():
        for points, labels in val_loader:
            points, labels = points.to(DEVICE), labels.to(DEVICE)

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

    ensemble_acc = 100. * correct / total

    print("\n結果:")
    print("-" * 50)
    for r in results:
        print(f"  {r['name']}: {r['val_acc']:.2f}%")
    print("-" * 50)
    print(f"  個別モデル平均: {np.mean([r['val_acc'] for r in results]):.2f}%")
    print(f"  アンサンブル:   {ensemble_acc:.2f}%")
    print("=" * 70)

    # 結果保存
    summary = {
        'dataset': str(DATASET_DIR),
        'total_samples': len(full_dataset),
        'train_samples': len(train_idx),
        'val_samples': len(val_idx),
        'class_weights': class_weights,
        'models': results,
        'ensemble_accuracy': ensemble_acc,
        'device': str(DEVICE)
    }

    with open(MODEL_DIR / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n保存先: {MODEL_DIR}")


if __name__ == "__main__":
    main()
