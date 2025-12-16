"""
多様な家具3Dモデル生成スクリプト
- 棚 (shelf)
- キャビネット (cabinet)
- デスク (desk)
- ソファ (sofa)
- スツール (stool)
- ベンチ (bench)
"""
import numpy as np
from pathlib import Path
import random

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "dataset"


def save_obj(vertices, faces, filepath):
    """OBJファイルとして保存"""
    with open(filepath, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {' '.join(str(i+1) for i in face)}\n")


def create_box(center, size):
    """直方体を生成"""
    cx, cy, cz = center
    sx, sy, sz = size

    vertices = [
        [cx - sx/2, cy - sy/2, cz - sz/2],
        [cx + sx/2, cy - sy/2, cz - sz/2],
        [cx + sx/2, cy + sy/2, cz - sz/2],
        [cx - sx/2, cy + sy/2, cz - sz/2],
        [cx - sx/2, cy - sy/2, cz + sz/2],
        [cx + sx/2, cy - sy/2, cz + sz/2],
        [cx + sx/2, cy + sy/2, cz + sz/2],
        [cx - sx/2, cy + sy/2, cz + sz/2],
    ]

    faces = [
        [0, 1, 2, 3],  # 前
        [4, 7, 6, 5],  # 後
        [0, 4, 5, 1],  # 下
        [2, 6, 7, 3],  # 上
        [0, 3, 7, 4],  # 左
        [1, 5, 6, 2],  # 右
    ]

    return np.array(vertices), faces


def create_cylinder(center, radius, height, segments=16):
    """円柱を生成"""
    cx, cy, cz = center
    vertices = []
    faces = []

    # 上下の円
    for y_offset in [0, height]:
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            x = cx + radius * np.cos(angle)
            z = cz + radius * np.sin(angle)
            vertices.append([x, cy + y_offset, z])

    # 中心点
    vertices.append([cx, cy, cz])
    vertices.append([cx, cy + height, cz])

    bottom_center = len(vertices) - 2
    top_center = len(vertices) - 1

    # 側面
    for i in range(segments):
        next_i = (i + 1) % segments
        faces.append([i, next_i, segments + next_i, segments + i])

    # 上下の面
    for i in range(segments):
        next_i = (i + 1) % segments
        faces.append([bottom_center, next_i, i])
        faces.append([top_center, segments + i, segments + next_i])

    return np.array(vertices), faces


def merge_meshes(meshes):
    """複数のメッシュを結合"""
    all_vertices = []
    all_faces = []
    vertex_offset = 0

    for vertices, faces in meshes:
        all_vertices.extend(vertices)
        for face in faces:
            all_faces.append([i + vertex_offset for i in face])
        vertex_offset += len(vertices)

    return np.array(all_vertices), all_faces


# =============================================================================
# 家具生成関数
# =============================================================================

def generate_shelf(stable=True):
    """棚を生成"""
    meshes = []

    # パラメータ
    width = random.uniform(0.6, 1.2)
    depth = random.uniform(0.25, 0.4)
    height = random.uniform(1.0, 1.8)
    num_shelves = random.randint(3, 5)
    thickness = 0.02

    if stable:
        # 安定: 幅広い底面、低重心
        base_height = random.uniform(0.05, 0.1)
    else:
        # 不安定: 細い脚、高重心
        base_height = 0
        height = random.uniform(1.5, 2.2)  # より高く
        width = random.uniform(0.3, 0.5)   # より狭く

    # 側板（左右）
    for x_pos in [-width/2 + thickness/2, width/2 - thickness/2]:
        v, f = create_box([x_pos, height/2, 0], [thickness, height, depth])
        meshes.append((v, f))

    # 棚板
    shelf_spacing = height / (num_shelves + 1)
    for i in range(num_shelves + 1):
        y_pos = i * shelf_spacing + thickness/2
        v, f = create_box([0, y_pos, 0], [width, thickness, depth])
        meshes.append((v, f))

    # 背板
    v, f = create_box([0, height/2, -depth/2 + thickness/2], [width, height, thickness])
    meshes.append((v, f))

    # 不安定版: 重心を上に偏らせる
    if not stable:
        # 上部に重い物体を追加
        v, f = create_box([0, height * 0.8, 0], [width * 0.8, 0.15, depth * 0.8])
        meshes.append((v, f))

    return merge_meshes(meshes)


def generate_cabinet(stable=True):
    """キャビネットを生成"""
    meshes = []

    width = random.uniform(0.4, 0.8)
    depth = random.uniform(0.35, 0.5)
    height = random.uniform(0.6, 1.0)
    leg_height = random.uniform(0.05, 0.15)

    if not stable:
        # 不安定: 細長い脚、片側に重心
        leg_height = random.uniform(0.2, 0.35)
        height = random.uniform(0.8, 1.2)

    # 本体
    v, f = create_box([0, leg_height + height/2, 0], [width, height, depth])
    meshes.append((v, f))

    # 脚
    leg_size = 0.04
    for x in [-width/2 + leg_size, width/2 - leg_size]:
        for z in [-depth/2 + leg_size, depth/2 - leg_size]:
            v, f = create_box([x, leg_height/2, z], [leg_size, leg_height, leg_size])
            meshes.append((v, f))

    # 不安定版: 開いた引き出し
    if not stable:
        drawer_depth = depth * 0.7
        v, f = create_box([0, leg_height + height * 0.7, depth/2 + drawer_depth/2],
                         [width * 0.8, 0.15, drawer_depth])
        meshes.append((v, f))

    return merge_meshes(meshes)


def generate_desk(stable=True):
    """デスクを生成"""
    meshes = []

    width = random.uniform(1.0, 1.6)
    depth = random.uniform(0.5, 0.8)
    height = random.uniform(0.7, 0.8)
    top_thickness = 0.03
    leg_size = random.uniform(0.04, 0.06)

    if not stable:
        # 不安定: 3本脚、または非対称
        leg_positions = [
            [-width/2 + leg_size, -depth/2 + leg_size],
            [width/2 - leg_size, -depth/2 + leg_size],
            [0, depth/2 - leg_size],  # 1本だけ後ろ
        ]
    else:
        # 安定: 4本脚
        leg_positions = [
            [-width/2 + leg_size, -depth/2 + leg_size],
            [width/2 - leg_size, -depth/2 + leg_size],
            [-width/2 + leg_size, depth/2 - leg_size],
            [width/2 - leg_size, depth/2 - leg_size],
        ]

    # 天板
    v, f = create_box([0, height - top_thickness/2, 0], [width, top_thickness, depth])
    meshes.append((v, f))

    # 脚
    leg_height = height - top_thickness
    for x, z in leg_positions:
        v, f = create_box([x, leg_height/2, z], [leg_size, leg_height, leg_size])
        meshes.append((v, f))

    return merge_meshes(meshes)


def generate_sofa(stable=True):
    """ソファを生成"""
    meshes = []

    width = random.uniform(1.5, 2.5)
    depth = random.uniform(0.7, 0.9)
    seat_height = random.uniform(0.35, 0.45)
    back_height = random.uniform(0.35, 0.5)
    arm_width = random.uniform(0.1, 0.2)

    if not stable:
        # 不安定: 高い背もたれ、狭い座面
        back_height = random.uniform(0.7, 0.9)
        depth = random.uniform(0.4, 0.5)

    # 座面
    v, f = create_box([0, seat_height/2, 0], [width, seat_height, depth])
    meshes.append((v, f))

    # 背もたれ
    v, f = create_box([0, seat_height + back_height/2, -depth/2 + 0.1],
                     [width, back_height, 0.15])
    meshes.append((v, f))

    # アームレスト
    for x_mult in [-1, 1]:
        x_pos = x_mult * (width/2 - arm_width/2)
        v, f = create_box([x_pos, seat_height + 0.15, 0],
                         [arm_width, 0.3, depth])
        meshes.append((v, f))

    return merge_meshes(meshes)


def generate_stool(stable=True):
    """スツールを生成"""
    meshes = []

    seat_radius = random.uniform(0.15, 0.25)
    height = random.uniform(0.45, 0.75)

    if stable:
        # 安定: 4本脚または太い中央脚
        if random.random() < 0.5:
            # 4本脚
            leg_size = 0.025
            for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
                x = (seat_radius - leg_size) * np.cos(angle)
                z = (seat_radius - leg_size) * np.sin(angle)
                v, f = create_box([x, height/2, z], [leg_size, height, leg_size])
                meshes.append((v, f))
        else:
            # 太い中央脚
            v, f = create_cylinder([0, 0, 0], seat_radius * 0.4, height)
            meshes.append((v, f))
    else:
        # 不安定: 3本脚、または細い中央脚
        if random.random() < 0.5:
            # 3本脚
            leg_size = 0.02
            for angle in [0, 2*np.pi/3, 4*np.pi/3]:
                x = (seat_radius - leg_size) * np.cos(angle)
                z = (seat_radius - leg_size) * np.sin(angle)
                v, f = create_box([x, height/2, z], [leg_size, height, leg_size])
                meshes.append((v, f))
        else:
            # 細い中央脚
            v, f = create_cylinder([0, 0, 0], seat_radius * 0.15, height)
            meshes.append((v, f))

    # 座面
    v, f = create_cylinder([0, height, 0], seat_radius, 0.03)
    meshes.append((v, f))

    return merge_meshes(meshes)


def generate_bench(stable=True):
    """ベンチを生成"""
    meshes = []

    width = random.uniform(1.0, 1.8)
    depth = random.uniform(0.3, 0.45)
    height = random.uniform(0.4, 0.5)
    top_thickness = 0.04
    leg_size = 0.05

    if not stable:
        # 不安定: 片側に傾斜、または細い脚
        leg_size = 0.025
        height = random.uniform(0.5, 0.7)

    # 座面
    v, f = create_box([0, height - top_thickness/2, 0], [width, top_thickness, depth])
    meshes.append((v, f))

    # 脚
    leg_height = height - top_thickness
    if stable:
        # 板状の脚（両端）
        for x in [-width/2 + 0.05, width/2 - 0.05]:
            v, f = create_box([x, leg_height/2, 0], [0.04, leg_height, depth * 0.8])
            meshes.append((v, f))
    else:
        # 4本の細い脚
        for x in [-width/2 + leg_size, width/2 - leg_size]:
            for z in [-depth/2 + leg_size, depth/2 - leg_size]:
                v, f = create_box([x, leg_height/2, z], [leg_size, leg_height, leg_size])
                meshes.append((v, f))

    return merge_meshes(meshes)


# =============================================================================
# メイン生成関数
# =============================================================================

FURNITURE_GENERATORS = {
    'shelf': generate_shelf,
    'cabinet': generate_cabinet,
    'desk': generate_desk,
    'sofa': generate_sofa,
    'stool': generate_stool,
    'bench': generate_bench,
}


def generate_dataset(num_per_type=200):
    """データセットを生成"""
    stable_dir = OUTPUT_DIR / "stable"
    unstable_dir = OUTPUT_DIR / "unstable"
    stable_dir.mkdir(parents=True, exist_ok=True)
    unstable_dir.mkdir(parents=True, exist_ok=True)

    # 既存のファイル数を取得
    existing_counts = {}
    for furniture_type in FURNITURE_GENERATORS.keys():
        stable_count = len(list(stable_dir.glob(f"{furniture_type}_*.obj")))
        unstable_count = len(list(unstable_dir.glob(f"{furniture_type}_*.obj")))
        existing_counts[furniture_type] = {'stable': stable_count, 'unstable': unstable_count}

    print("=" * 60)
    print("家具データセット生成")
    print("=" * 60)

    for furniture_type, generator in FURNITURE_GENERATORS.items():
        print(f"\n{furniture_type}を生成中...")

        # 安定版
        start_idx = existing_counts[furniture_type]['stable']
        for i in range(num_per_type):
            try:
                vertices, faces = generator(stable=True)
                filepath = stable_dir / f"{furniture_type}_{start_idx + i:05d}.obj"
                save_obj(vertices, faces, filepath)
            except Exception as e:
                print(f"  Error (stable {i}): {e}")
        print(f"  Stable: {num_per_type}個生成")

        # 不安定版
        start_idx = existing_counts[furniture_type]['unstable']
        num_unstable = num_per_type // 3  # 不安定は少なめ
        for i in range(num_unstable):
            try:
                vertices, faces = generator(stable=False)
                filepath = unstable_dir / f"{furniture_type}_{start_idx + i:05d}.obj"
                save_obj(vertices, faces, filepath)
            except Exception as e:
                print(f"  Error (unstable {i}): {e}")
        print(f"  Unstable: {num_unstable}個生成")

    # 統計
    print("\n" + "=" * 60)
    print("生成完了 - データセット統計")
    print("=" * 60)

    total_stable = 0
    total_unstable = 0

    for furniture_type in list(FURNITURE_GENERATORS.keys()) + ['chair', 'table']:
        stable_count = len(list(stable_dir.glob(f"{furniture_type}_*.obj")))
        unstable_count = len(list(unstable_dir.glob(f"{furniture_type}_*.obj")))
        total_stable += stable_count
        total_unstable += unstable_count
        print(f"  {furniture_type:10s}: {stable_count:4d} stable, {unstable_count:4d} unstable")

    print(f"\n  合計: {total_stable + total_unstable} サンプル")
    print(f"    Stable: {total_stable}, Unstable: {total_unstable}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', '-n', type=int, default=200, help='各タイプの生成数')
    args = parser.parse_args()

    generate_dataset(num_per_type=args.num)
