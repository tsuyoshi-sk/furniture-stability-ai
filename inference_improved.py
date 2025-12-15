"""
改善版推論スクリプト
- Test-Time Augmentation (TTA)
- 最適閾値探索
- 物理特徴の補助判定
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, List, Dict, Tuple

SCRIPT_DIR = Path(__file__).parent
MODEL_PATH = SCRIPT_DIR / "ensemble_models_v2" / "model_3.pth"
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
    """
    点群から物理的安定性に関連する特徴を計算

    Returns:
        dict: 物理特徴
    """
    if len(points) == 0:
        return {'stability_score': 0.0}

    # 重心
    centroid = np.mean(points, axis=0)

    # Y軸（高さ）の範囲
    y_min = np.min(points[:, 1])
    y_max = np.max(points[:, 1])
    height = y_max - y_min

    # 重心の相対高さ（0=底面、1=最上部）
    if height > 0:
        relative_cog_height = (centroid[1] - y_min) / height
    else:
        relative_cog_height = 0.5

    # 底面付近の点（下位10%）
    y_threshold = y_min + height * 0.1
    base_points = points[points[:, 1] <= y_threshold]

    if len(base_points) >= 3:
        # 底面の広がり（XZ平面）
        base_spread_x = np.max(base_points[:, 0]) - np.min(base_points[:, 0])
        base_spread_z = np.max(base_points[:, 2]) - np.min(base_points[:, 2])
        base_area = base_spread_x * base_spread_z

        # 重心が底面の範囲内にあるか
        cog_in_base_x = np.min(base_points[:, 0]) <= centroid[0] <= np.max(base_points[:, 0])
        cog_in_base_z = np.min(base_points[:, 2]) <= centroid[2] <= np.max(base_points[:, 2])
        cog_over_base = cog_in_base_x and cog_in_base_z
    else:
        base_area = 0
        cog_over_base = False

    # 全体の広がり
    spread_x = np.max(points[:, 0]) - np.min(points[:, 0])
    spread_z = np.max(points[:, 2]) - np.min(points[:, 2])

    # アスペクト比（高さ / 底面の広がり）
    base_spread = max(base_spread_x, base_spread_z) if len(base_points) >= 3 else max(spread_x, spread_z)
    aspect_ratio = height / base_spread if base_spread > 0 else float('inf')

    # 安定性スコア（ヒューリスティック）
    # - 重心が低いほど安定
    # - 底面が広いほど安定
    # - 重心が底面の上にあるほど安定
    stability_score = 0.0

    # 重心の低さ（0.5以下なら加点）
    if relative_cog_height < 0.5:
        stability_score += (0.5 - relative_cog_height) * 2  # max 1.0

    # 底面の広さ
    if base_area > 0:
        stability_score += min(base_area / (spread_x * spread_z + 1e-6), 1.0) * 0.5

    # 重心が底面上
    if cog_over_base:
        stability_score += 0.5

    # アスペクト比（低いほど安定）
    if aspect_ratio < 2.0:
        stability_score += (2.0 - aspect_ratio) / 2.0 * 0.5

    return {
        'centroid': centroid.tolist(),
        'relative_cog_height': float(relative_cog_height),
        'base_area': float(base_area) if len(base_points) >= 3 else 0.0,
        'aspect_ratio': float(aspect_ratio),
        'cog_over_base': bool(cog_over_base),
        'stability_score': float(min(stability_score, 2.5)),  # 正規化
        'height': float(height),
        'num_base_points': len(base_points)
    }


# =============================================================================
# TTA（Test-Time Augmentation）
# =============================================================================

def apply_tta_transforms(points: np.ndarray) -> List[np.ndarray]:
    """
    TTAのための変換を適用

    Returns:
        list: 変換された点群のリスト
    """
    transforms = [points.copy()]  # オリジナル

    # Y軸回転（90度ずつ4方向）
    for angle in [np.pi/2, np.pi, 3*np.pi/2]:
        Ry = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ], dtype=np.float32)
        transforms.append(points @ Ry.T)

    # X軸ミラーリング
    mirrored = points.copy()
    mirrored[:, 0] = -mirrored[:, 0]
    transforms.append(mirrored)

    # Z軸ミラーリング
    mirrored_z = points.copy()
    mirrored_z[:, 2] = -mirrored_z[:, 2]
    transforms.append(mirrored_z)

    # 微小スケール変動
    for scale in [0.95, 1.05]:
        scaled = points.copy() * scale
        transforms.append(scaled)

    return transforms


# =============================================================================
# モデル定義
# =============================================================================

class PointNetWithLocalFeatures(nn.Module):
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


# =============================================================================
# 改善版予測器
# =============================================================================

class ImprovedStabilityPredictor:
    """改善版安定性予測器"""

    def __init__(self, model_path=None, device=None, threshold=0.5):
        self.device = device or DEVICE
        self.model_path = model_path or MODEL_PATH
        self.threshold = threshold  # stable判定の閾値
        self.model = None
        self._load_model()

    def _load_model(self):
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        config = checkpoint.get('config', {})
        feature_dim = config.get('feature_dim', 512)
        dropout = config.get('dropout', 0.4)

        self.model = StabilityClassifier(feature_dim=feature_dim, dropout=dropout)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        val_acc = checkpoint.get('val_acc', 'N/A')
        print(f"モデル読み込み完了: {self.model_path.name}")
        print(f"検証精度: {val_acc}%, 閾値: {self.threshold}")

    def predict_with_tta(self, points: np.ndarray, num_samples: int = 3) -> Dict:
        """
        TTAを使用した予測

        Args:
            points: 正規化済み点群 [N, 3]
            num_samples: 各変換でのサンプリング回数

        Returns:
            dict: 予測結果
        """
        # TTA変換を適用
        transformed_points = apply_tta_transforms(points)

        all_probs = []

        with torch.no_grad():
            for trans_pts in transformed_points:
                for _ in range(num_samples):
                    sampled = sample_points(trans_pts, NUM_POINTS)
                    sampled = torch.from_numpy(sampled).float().transpose(0, 1)
                    sampled = sampled.unsqueeze(0).to(self.device)

                    outputs = self.model(sampled)
                    probs = torch.softmax(outputs, dim=1)
                    all_probs.append(probs.cpu().numpy()[0])

        # 平均確率
        avg_probs = np.mean(all_probs, axis=0)

        # 閾値ベースの判定
        stable_prob = avg_probs[1]
        prediction = 1 if stable_prob >= self.threshold else 0

        return {
            'prediction': 'stable' if prediction == 1 else 'unstable',
            'prediction_label': prediction,
            'confidence': float(avg_probs[prediction]),
            'prob_unstable': float(avg_probs[0]),
            'prob_stable': float(avg_probs[1]),
            'num_tta_transforms': len(transformed_points),
            'num_total_samples': len(all_probs)
        }

    def predict_file(self, obj_path, use_tta=True, use_physics=True, num_samples=3) -> Dict:
        """
        OBJファイルから予測

        Args:
            obj_path: OBJファイルパス
            use_tta: TTAを使用するか
            use_physics: 物理特徴を補助的に使用するか
            num_samples: サンプリング回数
        """
        vertices = load_obj_vertices(obj_path)
        raw_vertices = vertices.copy()
        vertices = normalize_point_cloud(vertices)

        # ニューラルネット予測
        if use_tta:
            nn_result = self.predict_with_tta(vertices, num_samples)
        else:
            nn_result = self._predict_simple(vertices, num_samples)

        result = {
            'file': str(obj_path),
            'num_vertices': len(raw_vertices),
            **nn_result
        }

        # 物理特徴
        if use_physics:
            physics = compute_physics_features(raw_vertices)
            result['physics'] = physics

            # 物理スコアとNNスコアの統合（重み付き平均）
            physics_stable_score = physics['stability_score'] / 2.5  # 0-1に正規化
            nn_stable_score = nn_result['prob_stable']

            # 統合スコア（NN 60%, Physics 40%）- 最適化済み
            combined_score = 0.60 * nn_stable_score + 0.40 * physics_stable_score
            result['combined_score'] = float(combined_score)

            # 統合判定
            result['combined_prediction'] = 'stable' if combined_score >= self.threshold else 'unstable'

        return result

    def _predict_simple(self, points: np.ndarray, num_samples: int = 5) -> Dict:
        """TTA無しのシンプルな予測"""
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
        stable_prob = avg_probs[1]
        prediction = 1 if stable_prob >= self.threshold else 0

        return {
            'prediction': 'stable' if prediction == 1 else 'unstable',
            'prediction_label': prediction,
            'confidence': float(avg_probs[prediction]),
            'prob_unstable': float(avg_probs[0]),
            'prob_stable': float(avg_probs[1]),
            'num_total_samples': num_samples
        }

    def find_optimal_threshold(self, dataset_dir: Path, num_samples: int = 100) -> Tuple[float, Dict]:
        """
        最適閾値を探索

        Returns:
            tuple: (最適閾値, 評価結果)
        """
        print("最適閾値を探索中...")

        # データ収集
        stable_probs = []
        unstable_probs = []

        for label, folder in [('stable', 'stable'), ('unstable', 'unstable')]:
            folder_path = dataset_dir / folder
            if not folder_path.exists():
                continue

            obj_files = list(folder_path.glob("*.obj"))
            np.random.shuffle(obj_files)
            obj_files = obj_files[:num_samples]

            for obj_file in obj_files:
                try:
                    vertices = load_obj_vertices(obj_file)
                    vertices = normalize_point_cloud(vertices)
                    result = self.predict_with_tta(vertices, num_samples=2)

                    if label == 'stable':
                        stable_probs.append(result['prob_stable'])
                    else:
                        unstable_probs.append(result['prob_stable'])
                except Exception as e:
                    print(f"  Error: {obj_file.name}: {e}")

        print(f"  Stable: {len(stable_probs)}, Unstable: {len(unstable_probs)}")

        # 閾値探索
        best_threshold = 0.5
        best_accuracy = 0
        best_f1 = 0
        results = {}

        for threshold in np.arange(0.3, 0.8, 0.02):
            tp = sum(1 for p in stable_probs if p >= threshold)
            fn = len(stable_probs) - tp
            tn = sum(1 for p in unstable_probs if p < threshold)
            fp = len(unstable_probs) - tn

            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            results[threshold] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
            }

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_accuracy = accuracy

        print(f"\n最適閾値: {best_threshold:.2f}")
        print(f"  Accuracy: {best_accuracy:.2%}")
        print(f"  F1 Score: {best_f1:.2%}")

        return best_threshold, results


def evaluate_model(predictor: ImprovedStabilityPredictor, dataset_dir: Path,
                   use_tta: bool = True, use_physics: bool = False,
                   max_samples: int = 200) -> Dict:
    """モデルを評価"""
    results = {'stable': [], 'unstable': []}

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
                results[label].append(result)
            except Exception as e:
                results[label].append({'file': str(obj_file), 'error': str(e)})

    # 精度計算
    correct = 0
    total = 0
    tp = tn = fp = fn = 0

    for label, predictions in results.items():
        for pred in predictions:
            if 'error' in pred:
                continue
            total += 1
            pred_key = 'combined_prediction' if use_physics and 'combined_prediction' in pred else 'prediction'
            pred_label = pred[pred_key]

            if pred_label == label:
                correct += 1
                if label == 'stable':
                    tp += 1
                else:
                    tn += 1
            else:
                if label == 'stable':
                    fn += 1
                else:
                    fp += 1

    accuracy = correct / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total': total,
        'correct': correct,
        'confusion': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn},
        'detailed_results': results
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='改善版家具安定性予測')
    parser.add_argument('input', nargs='?', help='OBJファイルまたはディレクトリ')
    parser.add_argument('--threshold', '-t', type=float, default=0.28, help='判定閾値（デフォルト: 0.28）')
    parser.add_argument('--no-tta', action='store_true', help='TTAを無効化')
    parser.add_argument('--no-physics', action='store_true', help='物理特徴を無効化（デフォルト: 有効）')
    parser.add_argument('--samples', '-s', type=int, default=3, help='サンプリング回数')
    parser.add_argument('--find-threshold', action='store_true', help='最適閾値を探索')
    parser.add_argument('--evaluate', action='store_true', help='データセットで評価')
    parser.add_argument('--output', '-o', help='結果をJSONに保存')
    args = parser.parse_args()

    predictor = ImprovedStabilityPredictor(threshold=args.threshold)
    dataset_dir = SCRIPT_DIR / "dataset"

    # 最適閾値探索
    if args.find_threshold:
        best_threshold, threshold_results = predictor.find_optimal_threshold(dataset_dir)
        predictor.threshold = best_threshold

        if args.output:
            with open(args.output, 'w') as f:
                json.dump({'best_threshold': best_threshold, 'results': threshold_results}, f, indent=2)
        return

    # 評価モード
    if args.evaluate:
        print("\n評価中...")
        eval_result = evaluate_model(
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
            # detailed_resultsを除いて保存
            save_result = {k: v for k, v in eval_result.items() if k != 'detailed_results'}
            with open(args.output, 'w') as f:
                json.dump(save_result, f, indent=2)
        return

    # 単一ファイル/ディレクトリ予測
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
        print(f"  NN予測: {result['prediction']} ({result['prob_stable']:.2%})")

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
