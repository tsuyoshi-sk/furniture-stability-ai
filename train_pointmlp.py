"""
PointMLP 学習スクリプト
- シンプルなMLP構造で点群を処理
- Residual接続とGeometric Affine Module
- 軽量で高速
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
DATASET_DIR = SCRIPT_DIR / "dataset"
MODEL_DIR = SCRIPT_DIR / "models_pointmlp"

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
# PointMLP コンポーネント
# =============================================================================

class GeometricAffineModule(nn.Module):
    """Geometric Affine Module - 点群の幾何学的特徴を正規化"""
    def __init__(self, channel):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channel, 1))
        self.beta = nn.Parameter(torch.zeros(1, channel, 1))

    def forward(self, x):
        # x: [B, C, N]
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-6
        x = (x - mean) / std
        return self.alpha * x + self.beta


class LocalGrouper(nn.Module):
    """近傍点をグループ化"""
    def __init__(self, channel, groups, kneighbors):
        super().__init__()
        self.groups = groups
        self.kneighbors = kneighbors

        # グループ中心をランダムサンプリングで選択
        self.affine = GeometricAffineModule(channel)

    def forward(self, xyz, points):
        """
        xyz: [B, N, 3]
        points: [B, N, C]
        return: [B, groups, kneighbors, C+3]
        """
        B, N, C = points.shape
        S = self.groups

        # ランダムにグループ中心を選択
        idx = torch.randperm(N, device=xyz.device)[:S]
        center_xyz = xyz[:, idx, :]  # [B, S, 3]
        center_points = points[:, idx, :]  # [B, S, C]

        # 各中心からk近傍を取得
        dist = torch.cdist(center_xyz, xyz)  # [B, S, N]
        _, knn_idx = dist.topk(self.kneighbors, dim=-1, largest=False)  # [B, S, k]

        # 近傍点を収集
        knn_idx_flat = knn_idx.reshape(B, -1)  # [B, S*k]

        grouped_xyz = torch.gather(
            xyz, 1, knn_idx_flat.unsqueeze(-1).expand(-1, -1, 3)
        ).reshape(B, S, self.kneighbors, 3)  # [B, S, k, 3]

        grouped_points = torch.gather(
            points, 1, knn_idx_flat.unsqueeze(-1).expand(-1, -1, C)
        ).reshape(B, S, self.kneighbors, C)  # [B, S, k, C]

        # 相対座標
        grouped_xyz = grouped_xyz - center_xyz.unsqueeze(2)

        # 特徴を結合
        new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)  # [B, S, k, C+3]

        return center_xyz, new_points


class PreExtraction(nn.Module):
    """特徴事前抽出"""
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, 1),
            nn.BatchNorm1d(out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        # x: [B, N, C] -> [B, C, N] -> [B, out, N] -> [B, N, out]
        return self.mlp(x.transpose(1, 2)).transpose(1, 2)


class PosExtraction(nn.Module):
    """位置特徴抽出"""
    def __init__(self, channel):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(channel, channel, 1),
            nn.BatchNorm1d(channel),
            nn.ReLU(),
            nn.Conv1d(channel, channel, 1),
            nn.BatchNorm1d(channel),
            nn.ReLU()
        )
        self.affine = GeometricAffineModule(channel)

    def forward(self, x):
        # x: [B, S, k, C]
        B, S, k, C = x.shape
        x = x.reshape(B * S, k, C).transpose(1, 2)  # [B*S, C, k]
        x = self.mlp(x)  # [B*S, C, k]
        x = self.affine(x)
        x = x.max(dim=-1)[0]  # [B*S, C]
        return x.reshape(B, S, C)


class PointMLPBlock(nn.Module):
    """PointMLP ブロック"""
    def __init__(self, in_channel, out_channel, groups, kneighbors):
        super().__init__()
        self.pre = PreExtraction(in_channel, out_channel)
        self.grouper = LocalGrouper(out_channel, groups, kneighbors)
        self.pos = PosExtraction(out_channel + 3)

        # 出力チャネル調整
        self.out_mlp = nn.Sequential(
            nn.Conv1d(out_channel + 3, out_channel, 1),
            nn.BatchNorm1d(out_channel),
            nn.ReLU()
        )

    def forward(self, xyz, points):
        """
        xyz: [B, N, 3]
        points: [B, N, C]
        """
        # 特徴抽出
        points = self.pre(points)

        # グループ化
        new_xyz, grouped = self.grouper(xyz, points)

        # 位置抽出
        new_points = self.pos(grouped)  # [B, S, C+3]

        # チャネル調整
        new_points = self.out_mlp(new_points.transpose(1, 2)).transpose(1, 2)

        return new_xyz, new_points


class PointMLPClassifier(nn.Module):
    """PointMLP 分類モデル"""
    def __init__(self, num_classes=2, embed_dim=64):
        super().__init__()

        # 入力埋め込み
        self.embed = nn.Sequential(
            nn.Conv1d(3, embed_dim, 1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )

        # PointMLP blocks
        self.block1 = PointMLPBlock(embed_dim, 128, groups=256, kneighbors=24)
        self.block2 = PointMLPBlock(128, 256, groups=64, kneighbors=24)
        self.block3 = PointMLPBlock(256, 512, groups=16, kneighbors=24)

        # 分類ヘッド
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        x: [B, N, 3]
        """
        B, N, _ = x.shape
        xyz = x.clone()

        # 埋め込み
        points = self.embed(x.transpose(1, 2)).transpose(1, 2)  # [B, N, embed_dim]

        # PointMLP blocks
        xyz, points = self.block1(xyz, points)
        xyz, points = self.block2(xyz, points)
        xyz, points = self.block3(xyz, points)

        # グローバルプーリング
        x = points.max(dim=1)[0]  # [B, 512]

        # 分類
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
    print("PointMLP 学習")
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
    model = PointMLPClassifier(num_classes=2, embed_dim=64).to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"パラメータ数: {num_params:,}")

    # 損失関数・オプティマイザ
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
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

        for points, labels_batch in train_loader:
            points, labels_batch = points.to(DEVICE), labels_batch.to(DEVICE)

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
                points, labels_batch = points.to(DEVICE), labels_batch.to(DEVICE)
                outputs = model(points)
                _, predicted = outputs.max(1)
                val_total += labels_batch.size(0)
                val_correct += predicted.eq(labels_batch).sum().item()

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
    model_path = MODEL_DIR / "pointmlp_best.pth"
    torch.save({
        'model_state_dict': best_state,
        'val_acc': best_val_acc,
        'config': {
            'architecture': 'PointMLP',
            'embed_dim': 64
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
            points, labels_batch = points.to(DEVICE), labels_batch.to(DEVICE)
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
