"""
家具安定性判定AIモデル - 学習スクリプト
PointNet風アーキテクチャで3D点群から安定/不安定を分類

使用法: python3 train_stability_model.py
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

# 設定
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATASET_DIR = PROJECT_ROOT / "data/datasets/dataset"
MODEL_SAVE_PATH = PROJECT_ROOT / "models/stability_model.pth"

NUM_POINTS = 1024  # サンプリングする点の数
BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 0.001

# デバイス設定（Apple Silicon MPS優先）
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using Apple Silicon MPS")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using CUDA")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")


def load_obj_vertices(filepath):
    """OBJファイルから頂点座標を読み込む"""
    vertices = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices.append([x, y, z])
    return np.array(vertices, dtype=np.float32)


def normalize_point_cloud(points):
    """点群を正規化（中心を原点、最大距離を1に）"""
    centroid = np.mean(points, axis=0)
    points = points - centroid
    max_dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    if max_dist > 0:
        points = points / max_dist
    return points


def sample_points(points, num_points):
    """点群からnum_points個をサンプリング"""
    n = len(points)
    if n == 0:
        return np.zeros((num_points, 3), dtype=np.float32)
    if n >= num_points:
        indices = np.random.choice(n, num_points, replace=False)
    else:
        indices = np.random.choice(n, num_points, replace=True)
    return points[indices]


class FurnitureDataset(Dataset):
    """家具データセット"""

    def __init__(self, root_dir, num_points=1024, augment=False):
        self.num_points = num_points
        self.augment = augment
        self.samples = []

        # stable/unstable フォルダからサンプルを収集
        for label, folder in [(1, "stable"), (0, "unstable")]:
            folder_path = Path(root_dir) / folder
            if not folder_path.exists():
                continue

            for obj_file in folder_path.glob("*.obj"):
                json_file = obj_file.with_suffix('.json')
                self.samples.append({
                    'obj_path': str(obj_file),
                    'json_path': str(json_file) if json_file.exists() else None,
                    'label': label
                })

        print(f"データセット読み込み完了: {len(self.samples)} サンプル")
        stable_count = sum(1 for s in self.samples if s['label'] == 1)
        print(f"  Stable: {stable_count}, Unstable: {len(self.samples) - stable_count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # OBJから頂点を読み込み
        vertices = load_obj_vertices(sample['obj_path'])

        # 正規化
        vertices = normalize_point_cloud(vertices)

        # サンプリング
        points = sample_points(vertices, self.num_points)

        # データ拡張（学習時のみ）
        if self.augment:
            points = self._augment(points)

        # Tensor変換 (N, 3) -> (3, N) for PointNet
        points = torch.from_numpy(points).float().transpose(0, 1)
        label = torch.tensor(sample['label'], dtype=torch.long)

        return points, label

    def _augment(self, points):
        """強化データ拡張"""
        # 1. ランダム回転（3軸全て）
        # X軸回転
        theta_x = np.random.uniform(-0.1, 0.1)  # 小さな傾き
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(theta_x), -np.sin(theta_x)],
            [0, np.sin(theta_x), np.cos(theta_x)]
        ], dtype=np.float32)

        # Y軸回転（フル回転）
        theta_y = np.random.uniform(0, 2 * np.pi)
        Ry = np.array([
            [np.cos(theta_y), 0, np.sin(theta_y)],
            [0, 1, 0],
            [-np.sin(theta_y), 0, np.cos(theta_y)]
        ], dtype=np.float32)

        # Z軸回転
        theta_z = np.random.uniform(-0.1, 0.1)  # 小さな傾き
        Rz = np.array([
            [np.cos(theta_z), -np.sin(theta_z), 0],
            [np.sin(theta_z), np.cos(theta_z), 0],
            [0, 0, 1]
        ], dtype=np.float32)

        points = points @ Rx.T @ Ry.T @ Rz.T

        # 2. ランダムミラーリング（50%の確率でX軸反転）
        if np.random.random() > 0.5:
            points[:, 0] = -points[:, 0]

        # 3. ランダムスケール（異方性）
        scale = np.random.uniform(0.85, 1.15, size=3).astype(np.float32)
        points *= scale

        # 4. ランダムシフト
        shift = np.random.uniform(-0.1, 0.1, size=3).astype(np.float32)
        points += shift

        # 5. ランダムノイズ（強化）
        noise_level = np.random.uniform(0.01, 0.03)
        points += np.random.normal(0, noise_level, points.shape).astype(np.float32)

        # 6. ランダム点ドロップアウト（5-15%の点を削除して復元）
        if np.random.random() > 0.5:
            drop_ratio = np.random.uniform(0.05, 0.15)
            n_drop = int(len(points) * drop_ratio)
            drop_indices = np.random.choice(len(points), n_drop, replace=False)
            keep_indices = np.setdiff1d(np.arange(len(points)), drop_indices)
            # ドロップした点を残りの点からランダムに復元
            restore_indices = np.random.choice(keep_indices, n_drop, replace=True)
            points[drop_indices] = points[restore_indices]

        return points


class PointNetEncoder(nn.Module):
    """PointNet特徴抽出器"""

    def __init__(self):
        super().__init__()

        # 共有MLP層
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

    def forward(self, x):
        # x: (B, 3, N)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))

        # Global Max Pooling
        x = torch.max(x, dim=2)[0]  # (B, 512)

        return x


class StabilityClassifier(nn.Module):
    """安定性分類モデル"""

    def __init__(self):
        super().__init__()

        self.encoder = PointNetEncoder()

        # 分類ヘッド
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)  # 2クラス: stable, unstable

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)  # 0.3 -> 0.5に強化

    def forward(self, x):
        x = self.encoder(x)

        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


def train_epoch(model, loader, criterion, optimizer):
    """1エポックの学習"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for points, labels in loader:
        points = points.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(points)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, criterion):
    """評価"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for points, labels in loader:
            points = points.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(points)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100. * correct / total


def main():
    print("\n" + "=" * 60)
    print("家具安定性判定AI - 学習開始")
    print("=" * 60)

    # データセット読み込み
    dataset = FurnitureDataset(DATASET_DIR, num_points=NUM_POINTS, augment=True)

    if len(dataset) < 10:
        print("\nエラー: データセットが少なすぎます。")
        print("先に dataset_pipeline.py で100個以上のデータを生成してください。")
        print("  ./run_blender.sh dataset_pipeline.py -- 200")
        return

    # 学習/検証分割（80/20）
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # 検証データは拡張なし
    val_dataset.dataset.augment = False

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\n学習データ: {train_size}, 検証データ: {val_size}")
    print(f"バッチサイズ: {BATCH_SIZE}, エポック数: {EPOCHS}")
    print(f"デバイス: {DEVICE}")

    # モデル初期化
    model = StabilityClassifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # ラベルスムージング追加
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)  # L2正則化追加
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # パラメータ数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"モデルパラメータ数: {total_params:,}")

    print("\n" + "-" * 60)
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>10} | {'Val Acc':>9}")
    print("-" * 60)

    best_val_acc = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        scheduler.step()

        print(f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>8.2f}% | {val_loss:>10.4f} | {val_acc:>8.2f}%")

        # ベストモデル保存
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, MODEL_SAVE_PATH)

    print("-" * 60)
    print(f"\n学習完了! ベスト検証精度: {best_val_acc:.2f}%")
    print(f"モデル保存先: {MODEL_SAVE_PATH}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
