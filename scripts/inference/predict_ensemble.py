"""
アンサンブル推論スクリプト
複数モデルの多数決で高精度な安定性判定
"""
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
ENSEMBLE_DIR = PROJECT_ROOT / "models/ensemble_models"
NUM_POINTS = 1024

# デバイス設定
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


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


def load_ensemble_models():
    """アンサンブルモデルを読み込み"""
    models = []
    model_files = sorted(ENSEMBLE_DIR.glob("model_*.pth"))

    if not model_files:
        raise FileNotFoundError(f"モデルが見つかりません: {ENSEMBLE_DIR}")

    for model_path in model_files:
        model = StabilityClassifier().to(DEVICE)
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models.append(model)

    return models


def predict_ensemble(obj_path, models):
    """アンサンブル推論"""
    # 点群読み込み
    vertices = load_obj_vertices(obj_path)
    vertices = normalize_point_cloud(vertices)
    points = sample_points(vertices, NUM_POINTS)
    points = torch.from_numpy(points).float().transpose(0, 1).unsqueeze(0).to(DEVICE)

    # 各モデルで推論
    all_probs = []
    individual_preds = []

    with torch.no_grad():
        for model in models:
            outputs = model(points)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs)
            individual_preds.append(outputs.argmax(dim=1).item())

    # ソフト投票（確率の平均）
    avg_probs = torch.stack(all_probs).mean(dim=0)
    final_pred = avg_probs.argmax(dim=1).item()
    confidence = avg_probs[0][final_pred].item() * 100

    # ハード投票（多数決）
    hard_vote = 1 if sum(individual_preds) > len(individual_preds) / 2 else 0

    labels = ["Unstable", "Stable"]

    return {
        "prediction": labels[final_pred],
        "confidence": confidence,
        "probabilities": {
            "unstable": avg_probs[0][0].item() * 100,
            "stable": avg_probs[0][1].item() * 100
        },
        "individual_votes": [labels[p] for p in individual_preds],
        "hard_vote": labels[hard_vote],
        "soft_vote": labels[final_pred]
    }


def main():
    if len(sys.argv) < 2:
        print("使用法: python3 predict_ensemble.py <obj_file>")
        print("例: python3 predict_ensemble.py output_chair.obj")
        return

    obj_path = sys.argv[1]

    if not Path(obj_path).exists():
        print(f"エラー: ファイルが見つかりません: {obj_path}")
        return

    print(f"\n{'=' * 55}")
    print("アンサンブル推論 - 家具安定性判定")
    print(f"{'=' * 55}")
    print(f"入力: {obj_path}")

    try:
        models = load_ensemble_models()
        print(f"モデル数: {len(models)}")
    except FileNotFoundError as e:
        print(f"\nエラー: {e}")
        print("先に train_ensemble.py を実行してください。")
        return

    result = predict_ensemble(obj_path, models)

    print(f"\n{'─' * 55}")
    print(f"最終判定: {result['prediction']}")
    print(f"確信度:   {result['confidence']:.1f}%")
    print(f"{'─' * 55}")
    print(f"\n各モデルの投票:")
    for i, vote in enumerate(result['individual_votes']):
        print(f"  モデル {i+1}: {vote}")
    print(f"\nソフト投票（確率平均）: {result['soft_vote']}")
    print(f"ハード投票（多数決）:   {result['hard_vote']}")
    print(f"\n詳細確率:")
    print(f"  Unstable: {result['probabilities']['unstable']:.1f}%")
    print(f"  Stable:   {result['probabilities']['stable']:.1f}%")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
