"""
PointNet++ 学習スクリプト
- 階層的特徴学習（Set Abstraction）
- マルチスケール特徴抽出
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
DATASET_DIR = SCRIPT_DIR / "dataset"
MODEL_DIR = SCRIPT_DIR / "models_pointnet2"

NUM_POINTS = 1024
BATCH_SIZE = 32
EPOCHS = 150
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

        points = torch.from_numpy(points).float()
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
        scale = np.random.uniform(0.9, 1.1)
        points *= scale

        # ノイズ
        points += np.random.normal(0, 0.02, points.shape).astype(np.float32)

        return points


# =============================================================================
# PointNet++ コンポーネント
# =============================================================================

def square_distance(src, dst):
    """
    2つの点群間の距離行列を計算
    src: [B, N, C]
    dst: [B, M, C]
    return: [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def farthest_point_sample(xyz, npoint):
    """
    Farthest Point Sampling
    xyz: [B, N, 3]
    npoint: サンプリング数
    return: [B, npoint] インデックス
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Ball Query
    radius: 球の半径
    nsample: 各球内の最大サンプル数
    xyz: [B, N, 3] 全点群
    new_xyz: [B, S, 3] クエリ点
    return: [B, S, nsample] インデックス
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape

    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx


def index_points(points, idx):
    """
    インデックスに基づいて点を取得
    points: [B, N, C]
    idx: [B, S] or [B, S, K]
    return: [B, S, C] or [B, S, K, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


class PointNetSetAbstraction(nn.Module):
    """Set Abstraction Layer"""
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        xyz: [B, N, 3] 座標
        points: [B, N, D] 特徴（Noneの場合は座標のみ）
        return: new_xyz [B, S, 3], new_points [B, S, D']
        """
        if self.group_all:
            new_xyz, new_points = self._sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = self._sample_and_group(xyz, points)

        # [B, S, nsample, C] -> [B, C, S, nsample]
        new_points = new_points.permute(0, 3, 1, 2).contiguous()

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        # Max pooling
        new_points = torch.max(new_points, -1)[0]  # [B, C, S]
        new_points = new_points.permute(0, 2, 1).contiguous()  # [B, S, C]

        return new_xyz, new_points

    def _sample_and_group(self, xyz, points):
        B, N, C = xyz.shape
        S = self.npoint

        fps_idx = farthest_point_sample(xyz, S)
        new_xyz = index_points(xyz, fps_idx)

        idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)
        grouped_xyz -= new_xyz.view(B, S, 1, C)

        if points is not None:
            grouped_points = index_points(points, idx)
            new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
        else:
            new_points = grouped_xyz

        return new_xyz, new_points

    def _sample_and_group_all(self, xyz, points):
        B, N, C = xyz.shape
        new_xyz = torch.zeros(B, 1, C, device=xyz.device)
        grouped_xyz = xyz.view(B, 1, N, C)

        if points is not None:
            new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
        else:
            new_points = grouped_xyz

        return new_xyz, new_points


class PointNet2Classifier(nn.Module):
    """PointNet++ 分類モデル"""
    def __init__(self, num_classes=2):
        super().__init__()

        # Set Abstraction layers
        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32,
            in_channel=3, mlp=[64, 64, 128]
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64,
            in_channel=128 + 3, mlp=[128, 128, 256]
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=256 + 3, mlp=[256, 512, 1024],
            group_all=True
        )

        # Classifier
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        """
        x: [B, N, 3]
        """
        B, N, _ = x.shape

        # Set Abstraction
        l1_xyz, l1_points = self.sa1(x, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # Global feature
        x = l3_points.view(B, 1024)

        # Classifier
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        return x


# =============================================================================
# 学習
# =============================================================================

def train():
    set_seed(42)
    MODEL_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("PointNet++ 学習")
    print("=" * 70)

    # データセット
    full_dataset = FurnitureDataset(DATASET_DIR, NUM_POINTS)
    labels = full_dataset.get_labels()
    label_counts = Counter(labels)

    print(f"\nデータセット: {len(full_dataset)} サンプル")
    print(f"  Stable: {label_counts[1]}, Unstable: {label_counts[0]}")

    # クラス重み
    total = len(labels)
    weight_unstable = total / (2 * label_counts[0])
    weight_stable = total / (2 * label_counts[1])
    class_weights = torch.tensor([weight_unstable, weight_stable], device=DEVICE)
    print(f"クラス重み: {class_weights.tolist()}")

    # 学習/検証分割
    indices = list(range(len(full_dataset)))
    np.random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_indices = indices[:split]
    val_indices = indices[split:]

    train_dataset = FurnitureDataset(DATASET_DIR, NUM_POINTS, augment=True)
    train_dataset.samples = [full_dataset.samples[i] for i in train_indices]

    val_dataset = FurnitureDataset(DATASET_DIR, NUM_POINTS, augment=False)
    val_dataset.samples = [full_dataset.samples[i] for i in val_indices]

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"学習: {len(train_dataset)}, 検証: {len(val_dataset)}")
    print(f"デバイス: {DEVICE}")

    # モデル
    model = PointNet2Classifier(num_classes=2).to(DEVICE)

    # パラメータ数
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"パラメータ数: {num_params:,}")

    # 損失関数・オプティマイザ
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    print("\n" + "-" * 70)
    print("学習開始")
    print("-" * 70)

    best_val_acc = 0
    best_state = None
    patience = 30
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

        with torch.no_grad():
            for points, labels in val_loader:
                points, labels = points.to(DEVICE), labels.to(DEVICE)
                outputs = model(points)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: Train={train_acc:.1f}%, Val={val_acc:.1f}% (Best={best_val_acc:.1f}%)")

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
    model_path = MODEL_DIR / "pointnet2_best.pth"
    torch.save({
        'model_state_dict': best_state,
        'val_acc': best_val_acc,
        'config': {
            'architecture': 'PointNet++',
            'num_points': NUM_POINTS
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
        for points, labels in val_loader:
            points, labels = points.to(DEVICE), labels.to(DEVICE)
            outputs = model(points)
            _, predicted = outputs.max(1)

            for p, l in zip(predicted, labels):
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
