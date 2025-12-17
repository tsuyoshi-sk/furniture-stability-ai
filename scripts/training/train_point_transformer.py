"""
Point Transformer 学習スクリプト
- Self-Attention機構を点群に適用
- 位置エンコーディング付き
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
MODEL_DIR = PROJECT_ROOT / "models/models_transformer"

NUM_POINTS = 512  # メモリ節約のため削減
BATCH_SIZE = 16   # メモリ節約のため削減
EPOCHS = 150
LEARNING_RATE = 0.0005

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
# Point Transformer コンポーネント
# =============================================================================

class PositionalEncoding(nn.Module):
    """位置エンコーディング（点群座標をMLPで変換）"""
    def __init__(self, in_dim=3, out_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, pos):
        return self.mlp(pos)


def knn_indices(pos, k):
    """メモリ効率の良いk-NN（バッチ単位で処理）"""
    B, N, _ = pos.shape
    indices_list = []

    for b in range(B):
        # 各バッチを個別に処理してメモリ節約
        pos_b = pos[b]  # [N, 3]
        dist = torch.cdist(pos_b.unsqueeze(0), pos_b.unsqueeze(0)).squeeze(0)  # [N, N]
        _, idx = torch.topk(dist, k, dim=-1, largest=False)  # [N, k]
        indices_list.append(idx)

    return torch.stack(indices_list, dim=0)  # [B, N, k]


class PointTransformerBlockLight(nn.Module):
    """軽量版 Point Transformer Block（メモリ効率改善）"""
    def __init__(self, dim, k=8):
        super().__init__()
        self.dim = dim
        self.k = k

        # 簡略化されたアテンション
        self.linear_q = nn.Linear(dim, dim)
        self.linear_k = nn.Linear(dim, dim)
        self.linear_v = nn.Linear(dim, dim)

        self.pos_enc = nn.Sequential(
            nn.Linear(3, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        self.attn_fc = nn.Linear(dim, 1)
        self.out_linear = nn.Linear(dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x, pos):
        """
        x: [B, N, C]
        pos: [B, N, 3]
        """
        B, N, C = x.shape
        identity = x

        # k-NN（no_gradでメモリ節約）
        with torch.no_grad():
            idx = knn_indices(pos, self.k)  # [B, N, k]

        # Query, Key, Value
        q = self.linear_q(x)  # [B, N, C]
        k = self.linear_k(x)  # [B, N, C]
        v = self.linear_v(x)  # [B, N, C]

        # Gather neighbors efficiently
        idx_flat = idx.reshape(B, -1)  # [B, N*k]

        # k neighbor features
        k_gathered = torch.gather(
            k, 1, idx_flat.unsqueeze(-1).expand(-1, -1, C)
        ).reshape(B, N, self.k, C)  # [B, N, k, C]

        v_gathered = torch.gather(
            v, 1, idx_flat.unsqueeze(-1).expand(-1, -1, C)
        ).reshape(B, N, self.k, C)  # [B, N, k, C]

        # Position difference
        pos_gathered = torch.gather(
            pos, 1, idx_flat.unsqueeze(-1).expand(-1, -1, 3)
        ).reshape(B, N, self.k, 3)  # [B, N, k, 3]

        pos_diff = pos.unsqueeze(2) - pos_gathered  # [B, N, k, 3]
        pos_encoding = self.pos_enc(pos_diff)  # [B, N, k, C]

        # Simple attention: q - k + pos_enc
        q_expanded = q.unsqueeze(2).expand(-1, -1, self.k, -1)  # [B, N, k, C]
        attn_input = q_expanded - k_gathered + pos_encoding  # [B, N, k, C]
        attn_weights = F.softmax(self.attn_fc(attn_input).squeeze(-1), dim=-1)  # [B, N, k]

        # Weighted sum of values
        out = torch.sum(attn_weights.unsqueeze(-1) * (v_gathered + pos_encoding), dim=2)  # [B, N, C]
        out = self.out_linear(out)

        # Residual + LayerNorm
        out = self.norm1(out + identity)

        # FFN
        out = self.norm2(out + self.ffn(out))

        return out


class PointTransformerClassifier(nn.Module):
    """Point Transformer 分類モデル（軽量版）"""
    def __init__(self, num_classes=2, dim=64, num_blocks=2, k=8):
        super().__init__()

        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(3, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        # Transformer blocks（軽量版を使用）
        self.blocks = nn.ModuleList([
            PointTransformerBlockLight(dim, k=k)
            for _ in range(num_blocks)
        ])

        # Global pooling + classifier
        self.pool_conv = nn.Conv1d(dim, dim * 2, 1)
        self.pool_bn = nn.BatchNorm1d(dim * 2)

        self.fc1 = nn.Linear(dim * 2, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        """
        x: [B, N, 3]
        """
        B, N, _ = x.shape
        pos = x.clone()

        # Embedding
        x = self.input_embed(x)  # [B, N, dim]

        # Transformer blocks
        for block in self.blocks:
            x = block(x, pos)

        # Global pooling
        x = x.transpose(1, 2)  # [B, dim, N]
        x = F.relu(self.pool_bn(self.pool_conv(x)))  # [B, dim*2, N]
        x = torch.max(x, dim=2)[0]  # [B, dim*2]

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
    print("Point Transformer 学習")
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

    # モデル（軽量版）
    model = PointTransformerClassifier(num_classes=2, dim=64, num_blocks=2, k=8).to(DEVICE)
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
    model_path = MODEL_DIR / "point_transformer_best.pth"
    torch.save({
        'model_state_dict': best_state,
        'val_acc': best_val_acc,
        'config': {
            'architecture': 'Point Transformer Light',
            'dim': 64,
            'num_blocks': 2,
            'k': 8,
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
