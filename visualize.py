#!/usr/bin/env python3
"""
家具安定性 3D可視化ツール
- OBJファイルの3D表示
- 予測結果のオーバーレイ
- 安定性特徴の可視化（重心、底面積など）
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
import argparse
import sys

# 推論モジュールをインポート
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from inference import FurnitureStabilityPredictor, load_obj_vertices, compute_physics_features


def load_obj_faces(filepath):
    """OBJファイルから面を読み込み"""
    faces = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('f '):
                parts = line.strip().split()[1:]
                # f v1/vt1/vn1 v2/vt2/vn2 ... の形式に対応
                face_indices = []
                for p in parts:
                    idx = int(p.split('/')[0]) - 1  # OBJは1始まり
                    face_indices.append(idx)
                faces.append(face_indices)
    return faces


def triangulate_face(face):
    """多角形を三角形に分割"""
    triangles = []
    for i in range(1, len(face) - 1):
        triangles.append([face[0], face[i], face[i + 1]])
    return triangles


def visualize_furniture(obj_path, predictor=None, save_path=None, show=True):
    """
    家具を3D可視化し、予測結果をオーバーレイ

    Args:
        obj_path: OBJファイルパス
        predictor: FurnitureStabilityPredictor インスタンス
        save_path: 保存先パス（指定時は画像保存）
        show: 表示するか
    """
    obj_path = Path(obj_path)

    # データ読み込み
    vertices = load_obj_vertices(obj_path)
    faces = load_obj_faces(obj_path)

    if len(vertices) == 0:
        print(f"Error: No vertices in {obj_path}")
        return None

    # 予測実行
    prediction = None
    if predictor:
        prediction = predictor.predict(obj_path, use_tta=True, num_samples=3)

    # 物理特徴計算
    physics = compute_physics_features(vertices)

    # 重心計算
    centroid = np.mean(vertices, axis=0)

    # 底面点抽出
    y_min = np.min(vertices[:, 1])
    y_max = np.max(vertices[:, 1])
    height = y_max - y_min
    y_threshold = y_min + height * 0.1
    base_points = vertices[vertices[:, 1] <= y_threshold]

    # Figure作成
    fig = plt.figure(figsize=(14, 6))

    # 3Dビュー
    ax1 = fig.add_subplot(121, projection='3d')

    # メッシュ描画
    triangles = []
    for face in faces:
        if len(face) >= 3:
            tris = triangulate_face(face)
            for tri in tris:
                if all(i < len(vertices) for i in tri):
                    triangles.append([vertices[i] for i in tri])

    if triangles:
        # 予測結果に応じた色
        if prediction:
            if prediction['prediction'] == 'stable':
                face_color = (0.3, 0.7, 0.3, 0.6)  # 緑
                edge_color = (0.2, 0.5, 0.2, 0.8)
            else:
                face_color = (0.8, 0.3, 0.3, 0.6)  # 赤
                edge_color = (0.6, 0.2, 0.2, 0.8)
        else:
            face_color = (0.5, 0.5, 0.8, 0.6)  # 青
            edge_color = (0.3, 0.3, 0.6, 0.8)

        mesh = Poly3DCollection(triangles, alpha=0.6)
        mesh.set_facecolor(face_color)
        mesh.set_edgecolor(edge_color)
        ax1.add_collection3d(mesh)

    # 重心をプロット
    ax1.scatter(*centroid, color='blue', s=100, marker='o', label='CoG (Center of Gravity)')

    # 重心から底面への垂直線
    ax1.plot([centroid[0], centroid[0]],
             [centroid[1], y_min],
             [centroid[2], centroid[2]],
             'b--', linewidth=2, label='CoG vertical')

    # 底面投影点
    ax1.scatter(centroid[0], y_min, centroid[2], color='cyan', s=80, marker='x', label='Base projection')

    # 底面領域をハイライト
    if len(base_points) >= 3:
        base_x = [np.min(base_points[:, 0]), np.max(base_points[:, 0])]
        base_z = [np.min(base_points[:, 2]), np.max(base_points[:, 2])]

        # 底面の四角形
        base_rect = [
            [base_x[0], y_min, base_z[0]],
            [base_x[1], y_min, base_z[0]],
            [base_x[1], y_min, base_z[1]],
            [base_x[0], y_min, base_z[1]]
        ]
        base_poly = Poly3DCollection([base_rect], alpha=0.3)
        base_poly.set_facecolor('yellow')
        base_poly.set_edgecolor('orange')
        ax1.add_collection3d(base_poly)

    # 軸設定
    all_coords = vertices
    max_range = np.max([
        np.max(all_coords[:, 0]) - np.min(all_coords[:, 0]),
        np.max(all_coords[:, 1]) - np.min(all_coords[:, 1]),
        np.max(all_coords[:, 2]) - np.min(all_coords[:, 2])
    ]) / 2

    mid_x = (np.max(all_coords[:, 0]) + np.min(all_coords[:, 0])) / 2
    mid_y = (np.max(all_coords[:, 1]) + np.min(all_coords[:, 1])) / 2
    mid_z = (np.max(all_coords[:, 2]) + np.min(all_coords[:, 2])) / 2

    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y (Height)')
    ax1.set_zlabel('Z')
    ax1.legend(loc='upper left', fontsize=8)

    # タイトル
    title = f"{obj_path.name}"
    if prediction:
        status = "STABLE" if prediction['prediction'] == 'stable' else "UNSTABLE"
        conf = prediction['confidence'] * 100
        title += f"\n{status} ({conf:.1f}%)"
    ax1.set_title(title, fontsize=12, fontweight='bold')

    # 情報パネル
    ax2 = fig.add_subplot(122)
    ax2.axis('off')

    info_text = f"""
    File: {obj_path.name}
    Furniture Type: {prediction['furniture_type'] if prediction else 'N/A'}
    Vertices: {len(vertices)}
    Faces: {len(faces)}

    -----------------------------
    Prediction Result
    -----------------------------
    Result: {prediction['prediction'].upper() if prediction else 'N/A'}
    Confidence: {prediction['confidence']*100:.1f}%
    Stable Prob: {prediction['prob_stable']*100:.1f}%

    -----------------------------
    Physics Features
    -----------------------------
    CoG Height: {physics['relative_cog_height']:.2f} (0=bottom, 1=top)
    Base Area: {physics['base_area']:.4f}
    Aspect Ratio: {physics['aspect_ratio']:.2f}
    CoG over Base: {'Yes' if physics['cog_over_base'] else 'No'}
    Stability Score: {physics['stability_score']:.2f} / 2.5

    -----------------------------
    Dimensions
    -----------------------------
    Width (X): {np.max(vertices[:,0]) - np.min(vertices[:,0]):.3f}
    Height (Y): {height:.3f}
    Depth (Z): {np.max(vertices[:,2]) - np.min(vertices[:,2]):.3f}
    """

    # 背景色を予測結果に応じて変更
    if prediction:
        if prediction['prediction'] == 'stable':
            ax2.set_facecolor((0.9, 1.0, 0.9))
        else:
            ax2.set_facecolor((1.0, 0.9, 0.9))

    ax2.text(0.05, 0.95, info_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return prediction


def visualize_comparison(obj_paths, predictor, save_path=None, show=True):
    """
    複数の家具を比較可視化
    """
    n = len(obj_paths)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(6 * cols, 5 * rows))

    for i, obj_path in enumerate(obj_paths):
        obj_path = Path(obj_path)

        if not obj_path.exists():
            print(f"Skip: {obj_path} not found")
            continue

        vertices = load_obj_vertices(obj_path)
        faces = load_obj_faces(obj_path)
        prediction = predictor.predict(obj_path, use_tta=True, num_samples=2)

        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')

        # メッシュ描画
        triangles = []
        for face in faces:
            if len(face) >= 3:
                tris = triangulate_face(face)
                for tri in tris:
                    if all(idx < len(vertices) for idx in tri):
                        triangles.append([vertices[idx] for idx in tri])

        if triangles:
            if prediction['prediction'] == 'stable':
                color = (0.3, 0.7, 0.3, 0.6)
            else:
                color = (0.8, 0.3, 0.3, 0.6)

            mesh = Poly3DCollection(triangles, alpha=0.6)
            mesh.set_facecolor(color)
            mesh.set_edgecolor((0.2, 0.2, 0.2, 0.3))
            ax.add_collection3d(mesh)

        # 軸設定
        max_range = np.max([
            np.max(vertices[:, 0]) - np.min(vertices[:, 0]),
            np.max(vertices[:, 1]) - np.min(vertices[:, 1]),
            np.max(vertices[:, 2]) - np.min(vertices[:, 2])
        ]) / 2

        mid = np.mean(vertices, axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        status = "STABLE" if prediction['prediction'] == 'stable' else "UNSTABLE"
        ax.set_title(f"{obj_path.name}\n{status} ({prediction['confidence']*100:.1f}%)",
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def create_sample_gallery(dataset_dir, predictor, output_dir, num_per_type=2):
    """
    各家具タイプのサンプルギャラリーを作成
    """
    from inference import SUPPORTED_FURNITURE

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ftype in SUPPORTED_FURNITURE:
        stable_files = list((dataset_dir / 'stable').glob(f'{ftype}_*.obj'))[:num_per_type]
        unstable_files = list((dataset_dir / 'unstable').glob(f'{ftype}_*.obj'))[:num_per_type]

        all_files = stable_files + unstable_files

        if all_files:
            save_path = output_dir / f"gallery_{ftype}.png"
            print(f"\nCreating {ftype} gallery...")
            visualize_comparison(all_files, predictor, save_path=save_path, show=False)

    print(f"\nGallery created: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Furniture Stability 3D Visualization')
    parser.add_argument('input', nargs='*', help='OBJ file(s)')
    parser.add_argument('--save', '-s', help='Save image to path')
    parser.add_argument('--no-show', action='store_true', help='Do not display (save only)')
    parser.add_argument('--compare', action='store_true', help='Comparison mode')
    parser.add_argument('--gallery', action='store_true', help='Create gallery for all types')
    parser.add_argument('--gallery-dir', default='visualizations', help='Gallery output directory')
    args = parser.parse_args()

    # 予測器初期化
    print("Loading model...")
    predictor = FurnitureStabilityPredictor()

    dataset_dir = SCRIPT_DIR / "dataset"

    # ギャラリーモード
    if args.gallery:
        create_sample_gallery(dataset_dir, predictor,
                             SCRIPT_DIR / args.gallery_dir,
                             num_per_type=2)
        return

    # 入力がない場合はサンプルを表示
    if not args.input:
        print("\nUsage:")
        print("  python3 visualize.py file.obj")
        print("  python3 visualize.py file1.obj file2.obj --compare")
        print("  python3 visualize.py file.obj --save output.png")
        print("  python3 visualize.py --gallery")

        # デモ: ランダムなサンプルを表示
        sample_files = list(dataset_dir.glob("**/*.obj"))[:1]
        if sample_files:
            print(f"\nDemo: {sample_files[0].name}")
            visualize_furniture(sample_files[0], predictor, show=not args.no_show)
        return

    # 入力パス解決
    input_paths = []
    for inp in args.input:
        path = Path(inp)
        if not path.is_absolute():
            path = Path.cwd() / path
        if path.exists():
            input_paths.append(path)
        else:
            print(f"Warning: {inp} not found")

    if not input_paths:
        print("Error: No valid files")
        return

    # 比較モード
    if args.compare or len(input_paths) > 1:
        visualize_comparison(input_paths, predictor,
                            save_path=args.save,
                            show=not args.no_show)
    else:
        visualize_furniture(input_paths[0], predictor,
                           save_path=args.save,
                           show=not args.no_show)


if __name__ == "__main__":
    main()
