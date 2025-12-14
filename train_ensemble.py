"""
アンサンブル学習スクリプト
複数モデルを異なるシードで訓練し、多数決で精度向上
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

SCRIPT_DIR = Path(__file__).parent
DATASET_DIR = SCRIPT_DIR / "dataset"
ENSEMBLE_DIR = SCRIPT_DIR / "ensemble_models"

NUM_POINTS = 1024
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
NUM_MODELS = 5  # アンサンブルするモデル数

# デバイス設定
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def set_seed(seed):
    """再現性のためのシード設定"""
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


class PointNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = torch.max(x, dim=2)[0]
        return x


class StabilityClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PointNetEncoder()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def train_single_model(train_loader, val_loader, model_id, seed):
    """1つのモデルを訓練"""
    set_seed(seed)

    model = StabilityClassifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    best_val_acc = 0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        # 訓練
        model.train()
        for points, labels in train_loader:
            points, labels = points.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(points)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

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
            best_state = model.state_dict().copy()

        if epoch % 20 == 0:
            print(f"  Model {model_id} Epoch {epoch}: Val Acc = {val_acc:.2f}%")

    # ベストモデルを保存
    model_path = ENSEMBLE_DIR / f"model_{model_id}.pth"
    torch.save({
        'model_state_dict': best_state,
        'val_acc': best_val_acc,
        'seed': seed
    }, model_path)

    return best_val_acc


def evaluate_ensemble(models, val_loader):
    """アンサンブル評価"""
    correct = 0
    total = 0

    for model in models:
        model.eval()

    with torch.no_grad():
        for points, labels in val_loader:
            points, labels = points.to(DEVICE), labels.to(DEVICE)

            # 各モデルの予測を収集
            all_probs = []
            for model in models:
                outputs = model(points)
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs)

            # 確率の平均（ソフト投票）
            avg_probs = torch.stack(all_probs).mean(dim=0)
            _, predicted = avg_probs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100. * correct / total


def main():
    print("\n" + "=" * 60)
    print("アンサンブル学習開始")
    print(f"モデル数: {NUM_MODELS}")
    print("=" * 60)

    # ディレクトリ作成
    ENSEMBLE_DIR.mkdir(exist_ok=True)

    # データセット読み込み
    full_dataset = FurnitureDataset(DATASET_DIR, num_points=NUM_POINTS, augment=True)
    print(f"\nデータセット: {len(full_dataset)} サンプル")

    # 学習/検証分割
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # 固定の検証セット
    set_seed(42)
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    val_dataset = Subset(full_dataset, val_indices)
    val_dataset.dataset.augment = False
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"学習データ: {train_size}, 検証データ: {val_size}")
    print(f"デバイス: {DEVICE}")

    # 各モデルを訓練
    individual_accs = []
    for i in range(NUM_MODELS):
        print(f"\n--- モデル {i+1}/{NUM_MODELS} 訓練中 ---")

        # 各モデルは異なるシードでデータをシャッフル
        seed = 100 + i * 10
        set_seed(seed)

        train_dataset = Subset(full_dataset, train_indices)
        train_dataset.dataset.augment = True
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        acc = train_single_model(train_loader, val_loader, i, seed)
        individual_accs.append(acc)
        print(f"  モデル {i+1} ベスト精度: {acc:.2f}%")

    # アンサンブル評価
    print("\n--- アンサンブル評価 ---")
    models = []
    for i in range(NUM_MODELS):
        model = StabilityClassifier().to(DEVICE)
        checkpoint = torch.load(ENSEMBLE_DIR / f"model_{i}.pth", map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        models.append(model)

    ensemble_acc = evaluate_ensemble(models, val_loader)

    print("\n" + "=" * 60)
    print("結果:")
    print("-" * 40)
    for i, acc in enumerate(individual_accs):
        print(f"  モデル {i+1}: {acc:.2f}%")
    print("-" * 40)
    print(f"  個別モデル平均: {np.mean(individual_accs):.2f}%")
    print(f"  アンサンブル:   {ensemble_acc:.2f}%")
    print("=" * 60)

    # 結果保存
    results = {
        "individual_accuracies": individual_accs,
        "mean_accuracy": float(np.mean(individual_accs)),
        "ensemble_accuracy": ensemble_acc,
        "num_models": NUM_MODELS,
        "dataset_size": len(full_dataset)
    }
    with open(ENSEMBLE_DIR / "results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nモデル保存先: {ENSEMBLE_DIR}")


if __name__ == "__main__":
    main()
