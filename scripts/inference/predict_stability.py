"""
家具安定性判定AI - 推論スクリプト
学習済みモデルで新しい3DモデルのSstable/Unstableを予測

使用法: python3 predict_stability.py path/to/model.obj
"""
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODEL_PATH = PROJECT_ROOT / "models/stability_model.pth"
NUM_POINTS = 1024


# デバイス設定
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


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
    """点群を正規化"""
    centroid = np.mean(points, axis=0)
    points = points - centroid
    max_dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    if max_dist > 0:
        points = points / max_dist
    return points


def sample_points(points, num_points):
    """点群からサンプリング"""
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
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def predict(obj_path):
    """OBJファイルの安定性を予測"""
    # モデル読み込み
    model = StabilityClassifier().to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 点群読み込み
    vertices = load_obj_vertices(obj_path)
    vertices = normalize_point_cloud(vertices)
    points = sample_points(vertices, NUM_POINTS)

    # Tensor変換
    points = torch.from_numpy(points).float().transpose(0, 1).unsqueeze(0).to(DEVICE)

    # 推論
    with torch.no_grad():
        outputs = model(points)
        probabilities = torch.softmax(outputs, dim=1)
        predicted = outputs.argmax(dim=1).item()

    labels = ["Unstable", "Stable"]
    confidence = probabilities[0][predicted].item() * 100

    return labels[predicted], confidence, probabilities[0].cpu().numpy()


def main():
    if len(sys.argv) < 2:
        print("使用法: python3 predict_stability.py <obj_file>")
        print("例: python3 predict_stability.py output_chair.obj")
        return

    obj_path = sys.argv[1]

    if not Path(obj_path).exists():
        print(f"エラー: ファイルが見つかりません: {obj_path}")
        return

    print(f"\n{'=' * 50}")
    print(f"家具安定性判定AI - 推論")
    print(f"{'=' * 50}")
    print(f"入力: {obj_path}")

    prediction, confidence, probs = predict(obj_path)

    print(f"\n結果: {prediction}")
    print(f"確信度: {confidence:.1f}%")
    print(f"\n詳細:")
    print(f"  Unstable: {probs[0]*100:.1f}%")
    print(f"  Stable:   {probs[1]*100:.1f}%")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()
