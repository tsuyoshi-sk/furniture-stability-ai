"""
高速版 PointNet++ 学習スクリプト
- ランダムサンプリング（FPSの代わり）
- k-NN（Ball Queryの代わり）
- 階層的特徴学習を維持
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
        theta = np.random.uniform(0, 2 * np.pi)
        Ry = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ], dtype=np.float32)
        points = points @ Ry.T
        if np.random.random() > 0.5:
            points[:, 0] = -points[:, 0]
        scale = np.random.uniform(0.9, 1.1)
        points *= scale
        points += np.random.normal(0, 0.02, points.shape).astype(np.float32)
        return points


# =============================================================================
# 高速版 PointNet++ コンポーネント
# =============================================================================

def knn(x, k):
    """
    k近傍探索
    x: [B, N, C]
    return: [B, N, k] インデックス
    """
    inner = -2 * torch.matmul(x, x.transpose(2, 1))
    xx = torch.sum(x ** 2, dim=2, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None):
    """
    グラフ特徴を取得
    x: [B, C, N]
    return: [B, 2C, N, k]
    """
    batch_size, num_dims, num_points = x.size()
    x = x.transpose(2, 1).contiguous()  # [B, N, C]

    if idx is None:
        idx = knn(x, k=k)

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.reshape(-1)

    x = x.reshape(batch_size * num_points, -1)
    feature = x[idx, :]
    feature = feature.reshape(batch_size, num_points, k, num_dims)
    x = x.reshape(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


class SetAbstractionFast(nn.Module):
    """高速版 Set Abstraction Layer（ランダムサンプリング + k-NN）"""
    def __init__(self, npoint, k, in_channel, mlp):
        super().__init__()
        self.npoint = npoint
        self.k = k

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel * 2  # edge feature
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points=None):
        """
        xyz: [B, N, 3]
        points: [B, N, D] or None
        return: new_xyz [B, S, 3], new_points [B, S, D']
        """
        B, N, _ = xyz.shape
        S = self.npoint

        # ランダムサンプリング
        idx = torch.randperm(N, device=xyz.device)[:S]
        idx = idx.unsqueeze(0).expand(B, -1)
        new_xyz = torch.gather(xyz, 1, idx.unsqueeze(-1).expand(-1, -1, 3))

        # 特徴を結合
        if points is not None:
            x = torch.cat([xyz, points], dim=-1)
        else:
            x = xyz

        x = x.transpose(2, 1).contiguous()  # [B, C, N]

        # k-NNでグラフ特徴取得
        x = get_graph_feature(x, k=self.k)  # [B, 2C, N, k]

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            x = F.relu(bn(conv(x)))

        x = x.max(dim=-1)[0]  # [B, C, N]

        # サンプリング位置の特徴を取得
        new_points = torch.gather(
            x.transpose(2, 1).contiguous(), 1,
            idx.unsqueeze(-1).expand(-1, -1, x.size(1))
        )  # [B, S, C]

        return new_xyz, new_points


class GlobalSetAbstraction(nn.Module):
    """グローバル特徴抽出"""
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        xyz: [B, N, 3]
        points: [B, N, D]
        return: [B, D']
        """
        x = torch.cat([xyz, points], dim=-1)  # [B, N, 3+D]
        x = x.transpose(2, 1).contiguous()  # [B, 3+D, N]

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            x = F.relu(bn(conv(x)))

        x = torch.max(x, dim=2)[0]  # [B, D']
        return x


class PointNet2Fast(nn.Module):
    """高速版 PointNet++ 分類モデル"""
    def __init__(self, num_classes=2):
        super().__init__()

        # Set Abstraction layers
        self.sa1 = SetAbstractionFast(npoint=512, k=16, in_channel=3, mlp=[64, 64, 128])
        self.sa2 = SetAbstractionFast(npoint=128, k=16, in_channel=3 + 128, mlp=[128, 128, 256])
        self.sa3 = GlobalSetAbstraction(in_channel=3 + 256, mlp=[256, 512, 1024])

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
        B = x.shape[0]

        # Set Abstraction
        l1_xyz, l1_points = self.sa1(x, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        x = self.sa3(l2_xyz, l2_points)

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
    print("高速版 PointNet++ 学習")
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

    # 分割
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
    model = PointNet2Fast(num_classes=2).to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"パラメータ数: {num_params:,}")

    # 損失関数・オプティマイザ
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    print("\n" + "-" * 70)
    print("学習開始")
    print("-" * 70)

    best_val_acc = 0
    best_state = None
    patience = 30
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
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
    model_path = MODEL_DIR / "pointnet2_fast_best.pth"
    torch.save({
        'model_state_dict': best_state,
        'val_acc': best_val_acc,
        'config': {
            'architecture': 'PointNet++ Fast',
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
