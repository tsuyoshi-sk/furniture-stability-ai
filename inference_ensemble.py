"""
Local + DGCNN アンサンブル推論スクリプト
- 最高性能の2モデル（Local 95.0%, DGCNN 94.4%）を組み合わせ
- TTA（Test-Time Augmentation）対応
- 物理特徴との統合
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, List, Dict, Tuple

SCRIPT_DIR = Path(__file__).parent
MODEL_DIR = SCRIPT_DIR / "ensemble_models_v2"
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
    """点群を正規化"""
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


# =============================================================================
# 物理特徴計算
# =============================================================================

def compute_physics_features(points: np.ndarray) -> Dict[str, float]:
    """点群から物理的安定性に関連する特徴を計算"""
    if len(points) == 0:
        return {'stability_score': 0.0}

    centroid = np.mean(points, axis=0)
    y_min = np.min(points[:, 1])
    y_max = np.max(points[:, 1])
    height = y_max - y_min

    if height > 0:
        relative_cog_height = (centroid[1] - y_min) / height
    else:
        relative_cog_height = 0.5

    y_threshold = y_min + height * 0.1
    base_points = points[points[:, 1] <= y_threshold]

    if len(base_points) >= 3:
        base_spread_x = np.max(base_points[:, 0]) - np.min(base_points[:, 0])
        base_spread_z = np.max(base_points[:, 2]) - np.min(base_points[:, 2])
        base_area = base_spread_x * base_spread_z
        cog_in_base_x = np.min(base_points[:, 0]) <= centroid[0] <= np.max(base_points[:, 0])
        cog_in_base_z = np.min(base_points[:, 2]) <= centroid[2] <= np.max(base_points[:, 2])
        cog_over_base = cog_in_base_x and cog_in_base_z
    else:
        base_area = 0
        base_spread_x = base_spread_z = 0
        cog_over_base = False

    spread_x = np.max(points[:, 0]) - np.min(points[:, 0])
    spread_z = np.max(points[:, 2]) - np.min(points[:, 2])
    base_spread = max(base_spread_x, base_spread_z) if len(base_points) >= 3 else max(spread_x, spread_z)
    aspect_ratio = height / base_spread if base_spread > 0 else float('inf')

    stability_score = 0.0
    if relative_cog_height < 0.5:
        stability_score += (0.5 - relative_cog_height) * 2
    if base_area > 0:
        stability_score += min(base_area / (spread_x * spread_z + 1e-6), 1.0) * 0.5
    if cog_over_base:
        stability_score += 0.5
    if aspect_ratio < 2.0:
        stability_score += (2.0 - aspect_ratio) / 2.0 * 0.5

    return {
        'relative_cog_height': float(relative_cog_height),
        'base_area': float(base_area) if len(base_points) >= 3 else 0.0,
        'aspect_ratio': float(aspect_ratio),
        'cog_over_base': bool(cog_over_base),
        'stability_score': float(min(stability_score, 2.5))
    }


# =============================================================================
# TTA（Test-Time Augmentation）
# =============================================================================

def apply_tta_transforms(points: np.ndarray) -> List[np.ndarray]:
    """TTAのための変換を適用"""
    transforms = [points.copy()]

    for angle in [np.pi/2, np.pi, 3*np.pi/2]:
        Ry = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ], dtype=np.float32)
        transforms.append(points @ Ry.T)

    mirrored = points.copy()
    mirrored[:, 0] = -mirrored[:, 0]
    transforms.append(mirrored)

    mirrored_z = points.copy()
    mirrored_z[:, 2] = -mirrored_z[:, 2]
    transforms.append(mirrored_z)

    for scale in [0.95, 1.05]:
        scaled = points.copy() * scale
        transforms.append(scaled)

    return transforms


# =============================================================================
# モデル定義
# =============================================================================

class PointNetWithLocalFeatures(nn.Module):
    """Local特徴付きPointNet"""
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


class DGCNNEncoder(nn.Module):
    """DGCNN風のエンコーダー（簡易版EdgeConv）"""
    def __init__(self, feature_dim=512, k=20):
        super().__init__()
        self.k = k

        self.conv1 = nn.Conv2d(6, 64, 1)
        self.conv2 = nn.Conv2d(128, 128, 1)
        self.conv3 = nn.Conv2d(256, 256, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv_out = nn.Conv1d(64 + 128 + 256, feature_dim, 1)
        self.bn_out = nn.BatchNorm1d(feature_dim)

    def knn(self, x):
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        idx = pairwise_distance.topk(k=self.k, dim=-1)[1]
        return idx

    def get_graph_feature(self, x, idx=None):
        batch_size, num_dims, num_points = x.size()
        if idx is None:
            idx = self.knn(x)

        idx_base = torch.arange(0, batch_size, device=x.device).reshape(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.reshape(-1)

        x = x.transpose(2, 1).contiguous()
        feature = x.reshape(batch_size * num_points, -1)[idx, :]
        feature = feature.reshape(batch_size, num_points, self.k, num_dims)
        x = x.reshape(batch_size, num_points, 1, num_dims).repeat(1, 1, self.k, 1)
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
        return feature

    def forward(self, x):
        batch_size = x.size(0)

        x1 = self.get_graph_feature(x)
        x1 = torch.relu(self.bn1(self.conv1(x1)))
        x1 = x1.max(dim=-1)[0]

        x2 = self.get_graph_feature(x1)
        x2 = torch.relu(self.bn2(self.conv2(x2)))
        x2 = x2.max(dim=-1)[0]

        x3 = self.get_graph_feature(x2)
        x3 = torch.relu(self.bn3(self.conv3(x3)))
        x3 = x3.max(dim=-1)[0]

        x = torch.cat([x1, x2, x3], dim=1)
        x = torch.relu(self.bn_out(self.conv_out(x)))
        x = torch.max(x, dim=2)[0]
        return x


class LocalClassifier(nn.Module):
    """Localモデル用分類器"""
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


class DGCNNClassifier(nn.Module):
    """DGCNNモデル用分類器"""
    def __init__(self, feature_dim=512, dropout=0.5, k=20):
        super().__init__()
        self.encoder = DGCNNEncoder(feature_dim, k)
        self.fc1 = nn.Linear(feature_dim, 256)  # DGCNNEncoderはfeature_dimを出力
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


# =============================================================================
# アンサンブル予測器
# =============================================================================

class EnsemblePredictor:
    """Local + DGCNN アンサンブル予測器"""

    def __init__(self, device=None, threshold=0.28):
        self.device = device or DEVICE
        self.threshold = threshold
        self.models = []
        self.weights = []
        self._load_models()

    def _load_models(self):
        """Local と DGCNN モデルをロード"""
        # Local model (model_3)
        local_path = MODEL_DIR / "model_3.pth"
        checkpoint = torch.load(local_path, map_location=self.device, weights_only=False)
        config = checkpoint.get('config', {})

        local_model = LocalClassifier(
            feature_dim=config.get('feature_dim', 512),
            dropout=config.get('dropout', 0.4)
        )
        local_model.load_state_dict(checkpoint['model_state_dict'])
        local_model.to(self.device)
        local_model.eval()

        self.models.append(('local', local_model))
        self.weights.append(config.get('weight', 0.5))
        print(f"Local モデル読み込み完了: Val Acc={checkpoint.get('val_acc', 'N/A')}%")

        # DGCNN model (model_4)
        dgcnn_path = MODEL_DIR / "model_4.pth"
        checkpoint = torch.load(dgcnn_path, map_location=self.device, weights_only=False)
        config = checkpoint.get('config', {})

        dgcnn_model = DGCNNClassifier(
            feature_dim=config.get('feature_dim', 512),
            dropout=config.get('dropout', 0.5),
            k=20
        )
        dgcnn_model.load_state_dict(checkpoint['model_state_dict'])
        dgcnn_model.to(self.device)
        dgcnn_model.eval()

        self.models.append(('dgcnn', dgcnn_model))
        self.weights.append(config.get('weight', 0.5))
        print(f"DGCNN モデル読み込み完了: Val Acc={checkpoint.get('val_acc', 'N/A')}%")

        # 最適重み（Local 80%, DGCNN 20%）- 検証精度 97.0%達成
        self.weights = [0.80, 0.20]
        print(f"アンサンブル重み: Local={self.weights[0]:.2%}, DGCNN={self.weights[1]:.2%}")

    def predict_with_tta(self, points: np.ndarray, num_samples: int = 3) -> Dict:
        """TTAを使用したアンサンブル予測"""
        transformed_points = apply_tta_transforms(points)

        all_probs = {name: [] for name, _ in self.models}

        with torch.no_grad():
            for trans_pts in transformed_points:
                for _ in range(num_samples):
                    sampled = sample_points(trans_pts, NUM_POINTS)
                    sampled = torch.from_numpy(sampled).float().transpose(0, 1)
                    sampled = sampled.unsqueeze(0).to(self.device)

                    for name, model in self.models:
                        outputs = model(sampled)
                        probs = torch.softmax(outputs, dim=1)
                        all_probs[name].append(probs.cpu().numpy()[0])

        # 各モデルの平均確率
        model_probs = {}
        for name, probs_list in all_probs.items():
            model_probs[name] = np.mean(probs_list, axis=0)

        # 重み付きアンサンブル
        ensemble_probs = np.zeros(2)
        for i, (name, _) in enumerate(self.models):
            ensemble_probs += self.weights[i] * model_probs[name]

        stable_prob = ensemble_probs[1]
        prediction = 1 if stable_prob >= self.threshold else 0

        return {
            'prediction': 'stable' if prediction == 1 else 'unstable',
            'prediction_label': prediction,
            'confidence': float(ensemble_probs[prediction]),
            'prob_unstable': float(ensemble_probs[0]),
            'prob_stable': float(ensemble_probs[1]),
            'model_probs': {name: {'stable': float(p[1]), 'unstable': float(p[0])}
                          for name, p in model_probs.items()},
            'num_tta_transforms': len(transformed_points),
            'num_total_samples': len(all_probs['local'])
        }

    def predict_file(self, obj_path, use_tta=True, use_physics=True, num_samples=3) -> Dict:
        """OBJファイルから予測"""
        vertices = load_obj_vertices(obj_path)
        raw_vertices = vertices.copy()
        vertices = normalize_point_cloud(vertices)

        if use_tta:
            nn_result = self.predict_with_tta(vertices, num_samples)
        else:
            nn_result = self._predict_simple(vertices, num_samples)

        result = {
            'file': str(obj_path),
            'num_vertices': len(raw_vertices),
            **nn_result
        }

        if use_physics:
            physics = compute_physics_features(raw_vertices)
            result['physics'] = physics

            physics_stable_score = physics['stability_score'] / 2.5
            nn_stable_score = nn_result['prob_stable']

            # 統合スコア（NN 60%, Physics 40%）
            combined_score = 0.60 * nn_stable_score + 0.40 * physics_stable_score
            result['combined_score'] = float(combined_score)
            result['combined_prediction'] = 'stable' if combined_score >= self.threshold else 'unstable'

        return result

    def _predict_simple(self, points: np.ndarray, num_samples: int = 5) -> Dict:
        """TTA無しのシンプルな予測"""
        all_probs = {name: [] for name, _ in self.models}

        with torch.no_grad():
            for _ in range(num_samples):
                sampled = sample_points(points, NUM_POINTS)
                sampled = torch.from_numpy(sampled).float().transpose(0, 1)
                sampled = sampled.unsqueeze(0).to(self.device)

                for name, model in self.models:
                    outputs = model(sampled)
                    probs = torch.softmax(outputs, dim=1)
                    all_probs[name].append(probs.cpu().numpy()[0])

        model_probs = {}
        for name, probs_list in all_probs.items():
            model_probs[name] = np.mean(probs_list, axis=0)

        ensemble_probs = np.zeros(2)
        for i, (name, _) in enumerate(self.models):
            ensemble_probs += self.weights[i] * model_probs[name]

        stable_prob = ensemble_probs[1]
        prediction = 1 if stable_prob >= self.threshold else 0

        return {
            'prediction': 'stable' if prediction == 1 else 'unstable',
            'prediction_label': prediction,
            'confidence': float(ensemble_probs[prediction]),
            'prob_unstable': float(ensemble_probs[0]),
            'prob_stable': float(ensemble_probs[1]),
            'model_probs': {name: {'stable': float(p[1]), 'unstable': float(p[0])}
                          for name, p in model_probs.items()},
            'num_total_samples': num_samples
        }


def evaluate_ensemble(predictor: EnsemblePredictor, dataset_dir: Path,
                      use_tta: bool = True, use_physics: bool = True,
                      max_samples: int = 200) -> Dict:
    """アンサンブルモデルを評価"""
    tp = tn = fp = fn = 0
    total = 0

    for label in ['stable', 'unstable']:
        folder_path = dataset_dir / label
        if not folder_path.exists():
            continue

        obj_files = list(folder_path.glob("*.obj"))
        np.random.shuffle(obj_files)
        obj_files = obj_files[:max_samples]

        print(f"  {label}: {len(obj_files)} files")

        for obj_file in obj_files:
            try:
                result = predictor.predict_file(obj_file, use_tta=use_tta,
                                                use_physics=use_physics, num_samples=2)
                total += 1

                pred_key = 'combined_prediction' if use_physics else 'prediction'
                pred_label = result[pred_key]

                if pred_label == label:
                    if label == 'stable':
                        tp += 1
                    else:
                        tn += 1
                else:
                    if label == 'stable':
                        fn += 1
                    else:
                        fp += 1
            except Exception as e:
                print(f"    Error: {obj_file.name}: {e}")

    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total': total,
        'confusion': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Local + DGCNN アンサンブル推論')
    parser.add_argument('input', nargs='?', help='OBJファイルまたはディレクトリ')
    parser.add_argument('--threshold', '-t', type=float, default=0.28, help='判定閾値')
    parser.add_argument('--no-tta', action='store_true', help='TTAを無効化')
    parser.add_argument('--no-physics', action='store_true', help='物理特徴を無効化')
    parser.add_argument('--samples', '-s', type=int, default=3, help='サンプリング回数')
    parser.add_argument('--evaluate', action='store_true', help='データセットで評価')
    parser.add_argument('--output', '-o', help='結果をJSONに保存')
    args = parser.parse_args()

    predictor = EnsemblePredictor(threshold=args.threshold)
    dataset_dir = SCRIPT_DIR / "dataset"

    if args.evaluate:
        print("\nアンサンブル評価中...")
        eval_result = evaluate_ensemble(
            predictor, dataset_dir,
            use_tta=not args.no_tta,
            use_physics=not args.no_physics
        )
        print(f"\n結果:")
        print(f"  Accuracy: {eval_result['accuracy']:.2%}")
        print(f"  Precision: {eval_result['precision']:.2%}")
        print(f"  Recall: {eval_result['recall']:.2%}")
        print(f"  F1 Score: {eval_result['f1']:.2%}")
        print(f"  Confusion: TP={eval_result['confusion']['tp']}, TN={eval_result['confusion']['tn']}, "
              f"FP={eval_result['confusion']['fp']}, FN={eval_result['confusion']['fn']}")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(eval_result, f, indent=2)
        return

    if not args.input:
        parser.print_help()
        return

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path

    if input_path.is_file():
        use_physics = not args.no_physics
        result = predictor.predict_file(
            input_path,
            use_tta=not args.no_tta,
            use_physics=use_physics,
            num_samples=args.samples
        )

        print(f"\n予測結果:")
        print(f"  ファイル: {Path(result['file']).name}")
        print(f"  アンサンブル予測: {result['prediction']} ({result['prob_stable']:.2%})")
        print(f"  - Local: {result['model_probs']['local']['stable']:.2%}")
        print(f"  - DGCNN: {result['model_probs']['dgcnn']['stable']:.2%}")

        if use_physics and 'combined_prediction' in result:
            print(f"  物理スコア: {result['physics']['stability_score']:.2f}/2.5")
            print(f"  統合予測: {result['combined_prediction']} ({result['combined_score']:.2%})")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

    elif input_path.is_dir():
        use_physics = not args.no_physics
        obj_files = list(input_path.glob("**/*.obj"))
        print(f"\n{len(obj_files)} 個のOBJファイルを処理中...")

        results = []
        for obj_file in obj_files:
            try:
                result = predictor.predict_file(
                    obj_file,
                    use_tta=not args.no_tta,
                    use_physics=use_physics,
                    num_samples=args.samples
                )
                results.append(result)
                pred_key = 'combined_prediction' if use_physics else 'prediction'
                print(f"  {obj_file.name}: {result[pred_key]}")
            except Exception as e:
                results.append({'file': str(obj_file), 'error': str(e)})

        pred_key = 'combined_prediction' if use_physics else 'prediction'
        stable_count = sum(1 for r in results if r.get(pred_key) == 'stable')
        unstable_count = sum(1 for r in results if r.get(pred_key) == 'unstable')

        print(f"\n結果: Stable={stable_count}, Unstable={unstable_count}")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    else:
        print(f"エラー: {input_path} が見つかりません")
        sys.exit(1)


if __name__ == "__main__":
    main()
