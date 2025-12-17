"""
Localモデル単体推論スクリプト
最高精度95.0%を達成したPointNetWithLocalFeaturesモデルを使用
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODEL_PATH = PROJECT_ROOT / "models/ensemble_models_v2" / "model_3.pth"

NUM_POINTS = 1024

# デバイス設定
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def load_obj_vertices(filepath):
    """OBJファイルから頂点を読み込み"""
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
    """点群をサンプリング"""
    n = len(points)
    if n == 0:
        return np.zeros((num_points, 3), dtype=np.float32)
    if n >= num_points:
        indices = np.random.choice(n, num_points, replace=False)
    else:
        indices = np.random.choice(n, num_points, replace=True)
    return points[indices]


class PointNetWithLocalFeatures(nn.Module):
    """ローカル＋グローバル特徴を結合するエンコーダー"""
    def __init__(self, feature_dim=512):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256 + 256, feature_dim, 1)
        self.bn4 = nn.BatchNorm1d(feature_dim)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        local_feat = torch.relu(self.bn3(self.conv3(x)))
        global_feat = torch.max(local_feat, dim=2, keepdim=True)[0]
        global_feat = global_feat.expand(-1, -1, local_feat.size(2))
        combined = torch.cat([local_feat, global_feat], dim=1)
        x = torch.relu(self.bn4(self.conv4(combined)))
        x = torch.max(x, dim=2)[0]
        return x


class StabilityClassifier(nn.Module):
    """安定性分類器"""
    def __init__(self, feature_dim=512, dropout=0.4):
        super().__init__()
        self.encoder = PointNetWithLocalFeatures(feature_dim)
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


class FurnitureStabilityPredictor:
    """家具安定性予測クラス"""

    def __init__(self, model_path=None, device=None):
        self.device = device or DEVICE
        self.model_path = model_path or MODEL_PATH
        self.model = None
        self._load_model()

    def _load_model(self):
        """モデルを読み込み"""
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        config = checkpoint.get('config', {})
        feature_dim = config.get('feature_dim', 512)
        dropout = config.get('dropout', 0.4)

        self.model = StabilityClassifier(feature_dim=feature_dim, dropout=dropout)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        val_acc = checkpoint.get('val_acc', 'N/A')
        print(f"モデル読み込み完了: {self.model_path}")
        print(f"検証精度: {val_acc}%")
        print(f"デバイス: {self.device}")

    def predict_file(self, obj_path, num_samples=5):
        """
        OBJファイルから安定性を予測

        Args:
            obj_path: OBJファイルのパス
            num_samples: サンプリング回数（平均を取る）

        Returns:
            dict: 予測結果
        """
        vertices = load_obj_vertices(obj_path)
        vertices = normalize_point_cloud(vertices)

        probs_list = []

        with torch.no_grad():
            for _ in range(num_samples):
                points = sample_points(vertices, NUM_POINTS)
                points = torch.from_numpy(points).float().transpose(0, 1)
                points = points.unsqueeze(0).to(self.device)

                outputs = self.model(points)
                probs = torch.softmax(outputs, dim=1)
                probs_list.append(probs.cpu().numpy()[0])

        avg_probs = np.mean(probs_list, axis=0)
        prediction = int(np.argmax(avg_probs))
        confidence = float(avg_probs[prediction])

        return {
            'file': str(obj_path),
            'prediction': 'stable' if prediction == 1 else 'unstable',
            'prediction_label': prediction,
            'confidence': confidence,
            'prob_unstable': float(avg_probs[0]),
            'prob_stable': float(avg_probs[1]),
            'num_vertices': len(vertices),
            'num_samples': num_samples
        }

    def predict_points(self, points, num_samples=5):
        """
        点群配列から安定性を予測

        Args:
            points: numpy配列 [N, 3]
            num_samples: サンプリング回数

        Returns:
            dict: 予測結果
        """
        points = normalize_point_cloud(points)

        probs_list = []

        with torch.no_grad():
            for _ in range(num_samples):
                sampled = sample_points(points, NUM_POINTS)
                sampled = torch.from_numpy(sampled).float().transpose(0, 1)
                sampled = sampled.unsqueeze(0).to(self.device)

                outputs = self.model(sampled)
                probs = torch.softmax(outputs, dim=1)
                probs_list.append(probs.cpu().numpy()[0])

        avg_probs = np.mean(probs_list, axis=0)
        prediction = int(np.argmax(avg_probs))
        confidence = float(avg_probs[prediction])

        return {
            'prediction': 'stable' if prediction == 1 else 'unstable',
            'prediction_label': prediction,
            'confidence': confidence,
            'prob_unstable': float(avg_probs[0]),
            'prob_stable': float(avg_probs[1]),
        }

    def predict_batch(self, obj_paths, num_samples=5):
        """
        複数ファイルをバッチ予測

        Args:
            obj_paths: OBJファイルパスのリスト
            num_samples: サンプリング回数

        Returns:
            list: 予測結果のリスト
        """
        results = []
        for path in obj_paths:
            try:
                result = self.predict_file(path, num_samples)
                results.append(result)
            except Exception as e:
                results.append({
                    'file': str(path),
                    'error': str(e)
                })
        return results


def main():
    """コマンドライン使用例"""
    import argparse

    parser = argparse.ArgumentParser(description='家具安定性予測（Localモデル）')
    parser.add_argument('input', nargs='?', help='OBJファイルまたはディレクトリのパス')
    parser.add_argument('--samples', '-s', type=int, default=5, help='サンプリング回数（デフォルト: 5）')
    parser.add_argument('--output', '-o', help='結果をJSONファイルに保存')
    parser.add_argument('--test', action='store_true', help='テストデータセットで評価')
    args = parser.parse_args()

    predictor = FurnitureStabilityPredictor()

    if args.test:
        # テストデータセットで評価
        dataset_dir = PROJECT_ROOT / "data/datasets/dataset"
        results = {'stable': [], 'unstable': []}

        for label, folder in [('stable', 'stable'), ('unstable', 'unstable')]:
            folder_path = dataset_dir / folder
            if folder_path.exists():
                obj_files = list(folder_path.glob("*.obj"))[:100]  # 各100個まで
                for obj_file in obj_files:
                    result = predictor.predict_file(obj_file, args.samples)
                    results[label].append(result)

        # 精度計算
        correct = 0
        total = 0
        for label, predictions in results.items():
            for pred in predictions:
                total += 1
                if pred['prediction'] == label:
                    correct += 1

        print(f"\n評価結果: {correct}/{total} = {100*correct/total:.2f}%")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"結果を保存: {args.output}")
        return

    if not args.input:
        parser.print_help()
        return

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path

    if input_path.is_file():
        # 単一ファイル
        result = predictor.predict_file(input_path, args.samples)
        print(f"\n予測結果:")
        print(f"  ファイル: {result['file']}")
        print(f"  予測: {result['prediction']} (信頼度: {result['confidence']:.2%})")
        print(f"  確率: stable={result['prob_stable']:.2%}, unstable={result['prob_unstable']:.2%}")
        print(f"  頂点数: {result['num_vertices']}")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

    elif input_path.is_dir():
        # ディレクトリ内の全OBJファイル
        obj_files = list(input_path.glob("**/*.obj"))
        print(f"\n{len(obj_files)} 個のOBJファイルを処理中...")

        results = predictor.predict_batch(obj_files, args.samples)

        stable_count = sum(1 for r in results if r.get('prediction') == 'stable')
        unstable_count = sum(1 for r in results if r.get('prediction') == 'unstable')

        print(f"\n結果サマリー:")
        print(f"  Stable: {stable_count}")
        print(f"  Unstable: {unstable_count}")

        for result in results:
            if 'error' in result:
                print(f"  [ERROR] {result['file']}: {result['error']}")
            else:
                print(f"  [{result['prediction']:8s}] {result['confidence']:.2%} - {Path(result['file']).name}")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n結果を保存: {args.output}")

    else:
        print(f"エラー: {input_path} が見つかりません")
        sys.exit(1)


if __name__ == "__main__":
    main()
