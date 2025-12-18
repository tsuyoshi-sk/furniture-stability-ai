#!/usr/bin/env python3
"""
家具安定性予測 推論スクリプト (v2)
- 8種類の家具に対応 (chair, table, shelf, cabinet, desk, sofa, stool, bench)
- 11,396サンプルで学習済み
- 検証精度: 97.11%
"""
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODEL_PATH = PROJECT_ROOT / "models/models_augmented" / "local_augmented_best.pth"
NUM_POINTS = 1024

# サポートする家具タイプ
SUPPORTED_FURNITURE = ['chair', 'table', 'shelf', 'cabinet', 'desk', 'sofa', 'stool', 'bench']

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
# TTA（Test-Time Augmentation）
# =============================================================================

def apply_tta_transforms(points: np.ndarray) -> List[np.ndarray]:
    """TTAのための変換を適用"""
    transforms = [points.copy()]

    # Y軸回転（90度ずつ）
    for angle in [np.pi/2, np.pi, 3*np.pi/2]:
        Ry = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ], dtype=np.float32)
        transforms.append(points @ Ry.T)

    # ミラーリング
    mirrored = points.copy()
    mirrored[:, 0] = -mirrored[:, 0]
    transforms.append(mirrored)

    mirrored_z = points.copy()
    mirrored_z[:, 2] = -mirrored_z[:, 2]
    transforms.append(mirrored_z)

    return transforms


# =============================================================================
# 物理特徴
# =============================================================================

def compute_physics_features(points: np.ndarray) -> Dict[str, float]:
    """点群から物理的安定性特徴を計算"""
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
        'base_area': float(base_area),
        'aspect_ratio': float(aspect_ratio),
        'cog_over_base': bool(cog_over_base),
        'stability_score': float(min(stability_score, 2.5))
    }


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


class StabilityClassifier(nn.Module):
    """安定性分類器"""
    def __init__(self, feature_dim=512, dropout=0.5):
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
# 予測クラス
# =============================================================================

class FurnitureStabilityPredictor:
    """家具安定性予測器"""

    def __init__(self, model_path: Optional[Path] = None, device=None, threshold: float = 0.5, verbose: bool = True):
        self.device = device or DEVICE
        self.model_path = model_path or MODEL_PATH
        self.threshold = threshold
        self.verbose = verbose
        self.model = None
        self._load_model()

    def _load_model(self):
        """モデルを読み込み"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        config = checkpoint.get('config', {})

        self.model = StabilityClassifier(
            feature_dim=config.get('feature_dim', 512),
            dropout=config.get('dropout', 0.5)
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        if self.verbose:
            val_acc = checkpoint.get('val_acc', 'N/A')
            print(f"モデル読み込み完了: {self.model_path.name}")
            print(f"  検証精度: {val_acc:.2f}%")
            print(f"  対応家具: {', '.join(SUPPORTED_FURNITURE)}")

    def predict(self, obj_path, use_tta: bool = True, num_samples: int = 3) -> Dict:
        """
        OBJファイルから安定性を予測

        Args:
            obj_path: OBJファイルパス
            use_tta: Test-Time Augmentationを使用
            num_samples: サンプリング回数

        Returns:
            予測結果の辞書
        """
        obj_path = Path(obj_path)
        vertices = load_obj_vertices(obj_path)
        raw_vertices = vertices.copy()
        vertices = normalize_point_cloud(vertices)

        # 予測
        if use_tta:
            result = self._predict_with_tta(vertices, num_samples)
        else:
            result = self._predict_simple(vertices, num_samples)

        # 物理特徴
        physics = compute_physics_features(raw_vertices)

        # 家具タイプを推定
        furniture_type = self._detect_furniture_type(obj_path.name)

        return {
            'file': str(obj_path),
            'furniture_type': furniture_type,
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'prob_stable': result['prob_stable'],
            'prob_unstable': result['prob_unstable'],
            'physics': physics,
            'num_vertices': len(raw_vertices)
        }

    def _predict_with_tta(self, points: np.ndarray, num_samples: int) -> Dict:
        """TTAを使用した予測"""
        transforms = apply_tta_transforms(points)
        all_probs = []

        with torch.no_grad():
            for trans_pts in transforms:
                for _ in range(num_samples):
                    sampled = sample_points(trans_pts, NUM_POINTS)
                    sampled = torch.from_numpy(sampled).float().transpose(0, 1)
                    sampled = sampled.unsqueeze(0).to(self.device)

                    outputs = self.model(sampled)
                    probs = torch.softmax(outputs, dim=1)
                    all_probs.append(probs.cpu().numpy()[0])

        avg_probs = np.mean(all_probs, axis=0)
        prediction = 1 if avg_probs[1] >= self.threshold else 0

        return {
            'prediction': 'stable' if prediction == 1 else 'unstable',
            'confidence': float(avg_probs[prediction]),
            'prob_stable': float(avg_probs[1]),
            'prob_unstable': float(avg_probs[0])
        }

    def _predict_simple(self, points: np.ndarray, num_samples: int) -> Dict:
        """シンプルな予測"""
        all_probs = []

        with torch.no_grad():
            for _ in range(num_samples):
                sampled = sample_points(points, NUM_POINTS)
                sampled = torch.from_numpy(sampled).float().transpose(0, 1)
                sampled = sampled.unsqueeze(0).to(self.device)

                outputs = self.model(sampled)
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy()[0])

        avg_probs = np.mean(all_probs, axis=0)
        prediction = 1 if avg_probs[1] >= self.threshold else 0

        return {
            'prediction': 'stable' if prediction == 1 else 'unstable',
            'confidence': float(avg_probs[prediction]),
            'prob_stable': float(avg_probs[1]),
            'prob_unstable': float(avg_probs[0])
        }

    def _detect_furniture_type(self, filename: str) -> str:
        """ファイル名から家具タイプを推定"""
        filename_lower = filename.lower()
        for ftype in SUPPORTED_FURNITURE:
            if ftype in filename_lower:
                return ftype
        return 'unknown'

    def predict_batch(self, obj_files: List[Path], use_tta: bool = True, num_samples: int = 2) -> List[Dict]:
        """複数ファイルをバッチ予測"""
        results = []
        for obj_file in obj_files:
            try:
                result = self.predict(obj_file, use_tta=use_tta, num_samples=num_samples)
                results.append(result)
            except Exception as e:
                results.append({'file': str(obj_file), 'error': str(e)})
        return results


def evaluate_dataset(predictor: FurnitureStabilityPredictor, dataset_dir: Path,
                     max_per_type: int = 50, use_tta: bool = True) -> Dict:
    """データセットを評価"""
    results = {}
    total_correct = 0
    total_count = 0

    for ftype in SUPPORTED_FURNITURE:
        stable_correct = 0
        stable_total = 0
        unstable_correct = 0
        unstable_total = 0

        # Stable
        stable_files = list((dataset_dir / 'stable').glob(f'{ftype}_*.obj'))[:max_per_type]
        for f in stable_files:
            try:
                result = predictor.predict(f, use_tta=use_tta, num_samples=2)
                stable_total += 1
                if result['prediction'] == 'stable':
                    stable_correct += 1
            except:
                pass

        # Unstable
        unstable_files = list((dataset_dir / 'unstable').glob(f'{ftype}_*.obj'))[:max_per_type]
        for f in unstable_files:
            try:
                result = predictor.predict(f, use_tta=use_tta, num_samples=2)
                unstable_total += 1
                if result['prediction'] == 'unstable':
                    unstable_correct += 1
            except:
                pass

        correct = stable_correct + unstable_correct
        total = stable_total + unstable_total
        accuracy = correct / total * 100 if total > 0 else 0

        results[ftype] = {
            'stable_acc': stable_correct / stable_total * 100 if stable_total > 0 else 0,
            'unstable_acc': unstable_correct / unstable_total * 100 if unstable_total > 0 else 0,
            'total_acc': accuracy,
            'samples': total
        }

        total_correct += correct
        total_count += total

    results['overall'] = {
        'accuracy': total_correct / total_count * 100 if total_count > 0 else 0,
        'total_samples': total_count
    }

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='家具安定性予測 (v2)')
    parser.add_argument('input', nargs='*', help='OBJファイル（複数可）またはディレクトリ')
    parser.add_argument('--threshold', '-t', type=float, default=0.5, help='判定閾値')
    parser.add_argument('--no-tta', action='store_true', help='TTAを無効化')
    parser.add_argument('--samples', '-s', type=int, default=3, help='サンプリング回数')
    parser.add_argument('--evaluate', action='store_true', help='データセットで評価')
    parser.add_argument('--output', '-o', help='結果をJSONに保存')
    args = parser.parse_args()

    predictor = FurnitureStabilityPredictor(threshold=args.threshold)
    dataset_dir = PROJECT_ROOT / "data/datasets/dataset"

    # 評価モード
    if args.evaluate:
        print("\n全家具タイプを評価中...")
        results = evaluate_dataset(predictor, dataset_dir, use_tta=not args.no_tta)

        print("\n" + "=" * 60)
        print("評価結果")
        print("=" * 60)

        for ftype in SUPPORTED_FURNITURE:
            r = results[ftype]
            print(f"  {ftype:10s}: Stable {r['stable_acc']:5.1f}%, Unstable {r['unstable_acc']:5.1f}%, Total: {r['total_acc']:.1f}%")

        print("-" * 60)
        print(f"  全体精度: {results['overall']['accuracy']:.1f}% ({results['overall']['total_samples']} samples)")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
        return

    # 単一ファイル/ディレクトリ予測
    if not args.input:
        parser.print_help()
        return

    # 入力パスを解決
    input_paths = []
    for inp in args.input:
        path = Path(inp)
        if not path.is_absolute():
            path = Path.cwd() / path
        input_paths.append(path)

    # 複数ファイルの場合
    if len(input_paths) > 1 or (len(input_paths) == 1 and input_paths[0].is_file() and len(args.input) > 1):
        obj_files = [p for p in input_paths if p.is_file() and p.suffix == '.obj']
        if not obj_files:
            print("エラー: 有効なOBJファイルがありません")
            sys.exit(1)

        print(f"\n{len(obj_files)} 個のOBJファイルを処理中...")
        results = predictor.predict_batch(obj_files, use_tta=not args.no_tta, num_samples=args.samples)

        for result in results:
            if 'error' in result:
                print(f"  {Path(result['file']).name}: エラー - {result['error']}")
            else:
                print(f"  {Path(result['file']).name}: {result['prediction']} ({result['prob_stable']*100:.1f}%)")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        return

    # 単一パスの場合
    input_path = input_paths[0]

    if input_path.is_file():
        result = predictor.predict(input_path, use_tta=not args.no_tta, num_samples=args.samples)

        print(f"\n予測結果:")
        print(f"  ファイル: {Path(result['file']).name}")
        print(f"  家具タイプ: {result['furniture_type']}")
        print(f"  予測: {result['prediction']} ({result['prob_stable']*100:.1f}% stable)")
        print(f"  信頼度: {result['confidence']*100:.1f}%")
        print(f"  物理スコア: {result['physics']['stability_score']:.2f}/2.5")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

    elif input_path.is_dir():
        obj_files = list(input_path.glob("**/*.obj"))
        print(f"\n{len(obj_files)} 個のOBJファイルを処理中...")

        results = predictor.predict_batch(obj_files, use_tta=not args.no_tta, num_samples=args.samples)

        stable_count = sum(1 for r in results if r.get('prediction') == 'stable')
        unstable_count = sum(1 for r in results if r.get('prediction') == 'unstable')

        print(f"\n結果: Stable={stable_count}, Unstable={unstable_count}")

        # タイプ別集計
        type_counts = {}
        for r in results:
            ftype = r.get('furniture_type', 'unknown')
            if ftype not in type_counts:
                type_counts[ftype] = {'stable': 0, 'unstable': 0}
            if r.get('prediction') == 'stable':
                type_counts[ftype]['stable'] += 1
            else:
                type_counts[ftype]['unstable'] += 1

        print("\nタイプ別:")
        for ftype, counts in sorted(type_counts.items()):
            print(f"  {ftype}: Stable={counts['stable']}, Unstable={counts['unstable']}")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    else:
        print(f"エラー: {input_path} が見つかりません")
        sys.exit(1)


if __name__ == "__main__":
    main()
