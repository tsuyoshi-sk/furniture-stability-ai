"""
家具データセット量産パイプライン
- ランダムなパラメータでテーブル・椅子を大量生成
- 各モデルの安定性を解析
- stable/unstable フォルダに自動振り分け

使用法: ./run_blender.sh dataset_pipeline.py -- [生成数]
例: ./run_blender.sh dataset_pipeline.py -- 100
"""
import bpy
import bmesh
import json
import os
import sys
import random
import math
import shutil
from mathutils import Vector

# 出力ディレクトリ
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")
STABLE_DIR = os.path.join(DATASET_DIR, "stable")
UNSTABLE_DIR = os.path.join(DATASET_DIR, "unstable")


def get_script_args():
    """引数を取得"""
    argv = sys.argv
    if "--" in argv:
        return argv[argv.index("--") + 1:]
    return []


def setup_directories():
    """ディレクトリ構造を作成"""
    os.makedirs(STABLE_DIR, exist_ok=True)
    os.makedirs(UNSTABLE_DIR, exist_ok=True)


def clear_scene():
    """シーンをクリア"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def create_table_with_params(params):
    """パラメータ指定でテーブルを生成"""
    objects = []

    table_width = params['width']
    table_depth = params['depth']
    table_height = params['height']
    top_thickness = params['top_thickness']
    leg_thickness = params['leg_thickness']
    leg_inset_x = params['leg_inset_x']  # 脚のX方向内側へのオフセット
    leg_inset_y = params['leg_inset_y']  # 脚のY方向内側へのオフセット
    top_offset_x = params.get('top_offset_x', 0)  # 天板のX方向オフセット
    top_offset_y = params.get('top_offset_y', 0)  # 天板のY方向オフセット

    # 天板を作成（オフセット可能）
    bpy.ops.mesh.primitive_cube_add(
        size=1,
        location=(top_offset_x, top_offset_y, table_height - top_thickness / 2)
    )
    top = bpy.context.active_object
    top.name = "TableTop"
    top.scale = (table_width / 2, table_depth / 2, top_thickness / 2)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    objects.append(top)

    # 4本の脚を作成
    leg_height = table_height - top_thickness
    leg_positions = [
        (table_width / 2 - leg_inset_x, table_depth / 2 - leg_inset_y),
        (-table_width / 2 + leg_inset_x, table_depth / 2 - leg_inset_y),
        (table_width / 2 - leg_inset_x, -table_depth / 2 + leg_inset_y),
        (-table_width / 2 + leg_inset_x, -table_depth / 2 + leg_inset_y),
    ]

    for i, (x, y) in enumerate(leg_positions):
        bpy.ops.mesh.primitive_cube_add(
            size=1,
            location=(x, y, leg_height / 2)
        )
        leg = bpy.context.active_object
        leg.name = f"Leg_{i+1}"
        leg.scale = (leg_thickness / 2, leg_thickness / 2, leg_height / 2)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        objects.append(leg)

    return objects


def create_chair_with_params(params):
    """パラメータ指定で椅子を生成"""
    objects = []

    seat_width = params['seat_width']
    seat_depth = params['seat_depth']
    seat_thickness = params['seat_thickness']
    seat_height = params['seat_height']
    leg_thickness = params['leg_thickness']
    backrest_height = params['backrest_height']
    backrest_thickness = params['backrest_thickness']
    backrest_angle = params['backrest_angle']
    seat_offset_x = params.get('seat_offset_x', 0)
    seat_offset_y = params.get('seat_offset_y', 0)

    # 座面を作成
    bpy.ops.mesh.primitive_cube_add(
        size=1,
        location=(seat_offset_x, seat_offset_y, seat_height + seat_thickness / 2)
    )
    seat = bpy.context.active_object
    seat.name = "Seat"
    seat.scale = (seat_width / 2, seat_depth / 2, seat_thickness / 2)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    objects.append(seat)

    # 4本の脚を作成
    leg_height = seat_height
    leg_inset = leg_thickness

    leg_positions = [
        (seat_width / 2 - leg_inset, seat_depth / 2 - leg_inset),
        (-seat_width / 2 + leg_inset, seat_depth / 2 - leg_inset),
        (seat_width / 2 - leg_inset, -seat_depth / 2 + leg_inset),
        (-seat_width / 2 + leg_inset, -seat_depth / 2 + leg_inset),
    ]

    for i, (x, y) in enumerate(leg_positions):
        bpy.ops.mesh.primitive_cube_add(
            size=1,
            location=(x, y, leg_height / 2)
        )
        leg = bpy.context.active_object
        leg.name = f"Leg_{i+1}"
        leg.scale = (leg_thickness / 2, leg_thickness / 2, leg_height / 2)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        objects.append(leg)

    # 背もたれを作成
    backrest_angle_rad = math.radians(backrest_angle)
    backrest_center_y = seat_depth / 2 - backrest_thickness / 2 + seat_offset_y
    backrest_center_z = seat_height + seat_thickness + backrest_height / 2
    backrest_offset_y = (backrest_height / 2) * math.sin(backrest_angle_rad)

    bpy.ops.mesh.primitive_cube_add(
        size=1,
        location=(seat_offset_x, backrest_center_y + backrest_offset_y, backrest_center_z)
    )
    backrest = bpy.context.active_object
    backrest.name = "Backrest"
    backrest.scale = (seat_width / 2, backrest_thickness / 2, backrest_height / 2)
    backrest.rotation_euler[0] = backrest_angle_rad
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    objects.append(backrest)

    return objects


def generate_chair_params(bias='balanced'):
    """椅子用ランダムパラメータを生成"""
    if bias == 'stable':
        seat_width = random.uniform(0.40, 0.50)
        seat_depth = random.uniform(0.40, 0.50)
        seat_thickness = random.uniform(0.03, 0.05)
        seat_height = random.uniform(0.40, 0.48)
        leg_thickness = random.uniform(0.04, 0.06)
        backrest_height = random.uniform(0.30, 0.45)
        backrest_thickness = random.uniform(0.02, 0.04)
        backrest_angle = random.uniform(0, 10)
        seat_offset_x = 0
        seat_offset_y = 0

    elif bias == 'very_unstable':
        seat_width = random.uniform(0.35, 0.45)
        seat_depth = random.uniform(0.35, 0.45)
        seat_thickness = random.uniform(0.03, 0.05)
        seat_height = random.uniform(0.42, 0.50)
        leg_thickness = random.uniform(0.03, 0.04)
        backrest_height = random.uniform(0.50, 0.70)  # 高い背もたれ
        backrest_thickness = random.uniform(0.03, 0.05)
        backrest_angle = random.uniform(20, 35)  # 大きく傾斜
        # 座面を大きくオフセット
        seat_offset_x = random.uniform(0.15, 0.25) * random.choice([-1, 1])
        seat_offset_y = random.uniform(0.10, 0.20) * random.choice([-1, 1])

    elif bias == 'unstable':
        seat_width = random.uniform(0.38, 0.48)
        seat_depth = random.uniform(0.38, 0.48)
        seat_thickness = random.uniform(0.03, 0.05)
        seat_height = random.uniform(0.42, 0.50)
        leg_thickness = random.uniform(0.03, 0.05)
        backrest_height = random.uniform(0.45, 0.60)
        backrest_thickness = random.uniform(0.03, 0.05)
        backrest_angle = random.uniform(12, 25)
        seat_offset_x = random.uniform(0.08, 0.15) * random.choice([-1, 1])
        seat_offset_y = random.uniform(0.05, 0.12) * random.choice([-1, 1])

    else:  # balanced
        seat_width = random.uniform(0.38, 0.52)
        seat_depth = random.uniform(0.38, 0.52)
        seat_thickness = random.uniform(0.03, 0.05)
        seat_height = random.uniform(0.38, 0.50)
        leg_thickness = random.uniform(0.03, 0.06)
        backrest_height = random.uniform(0.35, 0.55)
        backrest_thickness = random.uniform(0.02, 0.05)
        backrest_angle = random.uniform(0, 20)
        seat_offset_x = random.uniform(-0.12, 0.12)
        seat_offset_y = random.uniform(-0.10, 0.10)

    return {
        'seat_width': seat_width,
        'seat_depth': seat_depth,
        'seat_thickness': seat_thickness,
        'seat_height': seat_height,
        'leg_thickness': leg_thickness,
        'backrest_height': backrest_height,
        'backrest_thickness': backrest_thickness,
        'backrest_angle': backrest_angle,
        'seat_offset_x': seat_offset_x,
        'seat_offset_y': seat_offset_y
    }


def generate_table_params(bias='balanced'):
    """
    テーブル用ランダムパラメータを生成
    bias: 'balanced' - 安定/不安定が混在
          'stable' - 安定しやすいパラメータ
          'unstable' - 不安定になりやすいパラメータ
          'very_unstable' - 確実に不安定になるパラメータ
    """
    if bias == 'stable':
        # 安定しやすい: 幅広い天板、太い脚、脚が外側、天板中央
        width = random.uniform(0.8, 1.2)
        depth = random.uniform(0.6, 1.0)
        height = random.uniform(0.6, 0.8)
        top_thickness = random.uniform(0.03, 0.05)
        leg_thickness = random.uniform(0.06, 0.10)
        leg_inset_x = random.uniform(0.05, 0.10)
        leg_inset_y = random.uniform(0.05, 0.10)
        top_offset_x = 0
        top_offset_y = 0

    elif bias == 'very_unstable':
        # 確実に不安定: 天板を大きくずらす（重心が支持面の外に出る）
        width = random.uniform(1.0, 1.5)
        depth = random.uniform(0.6, 1.0)
        height = random.uniform(0.7, 0.9)
        top_thickness = random.uniform(0.04, 0.08)
        leg_thickness = random.uniform(0.04, 0.06)
        leg_inset_x = random.uniform(0.05, 0.15)
        leg_inset_y = random.uniform(0.05, 0.15)
        # 天板を大きくオフセット（重心が外に出る）
        top_offset_x = random.uniform(0.4, 0.7) * width * random.choice([-1, 1])
        top_offset_y = random.uniform(0.3, 0.5) * depth * random.choice([-1, 1])

    elif bias == 'unstable':
        # 不安定になりやすい: 天板を少しずらす
        width = random.uniform(1.0, 1.6)
        depth = random.uniform(0.5, 0.9)
        height = random.uniform(0.7, 1.0)
        top_thickness = random.uniform(0.04, 0.07)
        leg_thickness = random.uniform(0.04, 0.06)
        leg_inset_x = random.uniform(0.08, 0.20)
        leg_inset_y = random.uniform(0.08, 0.20)
        # 天板をオフセット
        top_offset_x = random.uniform(0.25, 0.45) * width * random.choice([-1, 1])
        top_offset_y = random.uniform(0.2, 0.35) * depth * random.choice([-1, 1])

    else:  # balanced
        width = random.uniform(0.6, 1.4)
        depth = random.uniform(0.5, 1.0)
        height = random.uniform(0.6, 0.9)
        top_thickness = random.uniform(0.03, 0.06)
        leg_thickness = random.uniform(0.04, 0.08)
        leg_inset_x = random.uniform(0.05, 0.15)
        leg_inset_y = random.uniform(0.05, 0.15)
        # 天板を少しオフセット（ランダム）
        top_offset_x = random.uniform(-0.3, 0.3) * width
        top_offset_y = random.uniform(-0.3, 0.3) * depth

    return {
        'width': width,
        'depth': depth,
        'height': height,
        'top_thickness': top_thickness,
        'leg_thickness': leg_thickness,
        'leg_inset_x': leg_inset_x,
        'leg_inset_y': leg_inset_y,
        'top_offset_x': top_offset_x,
        'top_offset_y': top_offset_y
    }


def calculate_volume_and_centroid(obj):
    """オブジェクトの体積と重心を計算"""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.transform(obj.matrix_world)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])

    total_volume = 0.0
    weighted_centroid = Vector((0.0, 0.0, 0.0))

    for face in bm.faces:
        if len(face.verts) == 3:
            v0, v1, v2 = [v.co for v in face.verts]
            signed_vol = v0.dot(v1.cross(v2)) / 6.0
            total_volume += signed_vol
            tetra_centroid = (v0 + v1 + v2) / 4.0
            weighted_centroid += tetra_centroid * signed_vol

    bm.free()

    volume = abs(total_volume)
    if volume > 1e-10:
        centroid = weighted_centroid / total_volume
    else:
        centroid = Vector((0.0, 0.0, 0.0))

    return volume, centroid


def calculate_combined_centroid(objects):
    """複数オブジェクトの合成重心"""
    total_volume = 0.0
    weighted_centroid = Vector((0.0, 0.0, 0.0))

    for obj in objects:
        if obj.type != 'MESH':
            continue
        volume, centroid = calculate_volume_and_centroid(obj)
        total_volume += volume
        weighted_centroid += centroid * volume

    if total_volume > 1e-10:
        return weighted_centroid / total_volume
    return Vector((0.0, 0.0, 0.0))


def find_support_polygon(objects):
    """支持基底面を計算"""
    ground_points = []
    min_z = float('inf')

    for obj in objects:
        if obj.type != 'MESH':
            continue
        for vert in obj.data.vertices:
            world_co = obj.matrix_world @ vert.co
            if world_co.z < min_z:
                min_z = world_co.z

    tolerance = 0.01
    for obj in objects:
        if obj.type != 'MESH':
            continue
        for vert in obj.data.vertices:
            world_co = obj.matrix_world @ vert.co
            if abs(world_co.z - min_z) < tolerance:
                ground_points.append((world_co.x, world_co.y))

    return ground_points


def compute_convex_hull_2d(points):
    """2D凸包計算"""
    if len(points) < 3:
        return points

    points = sorted(set(points))
    start = min(points, key=lambda p: (p[1], p[0]))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    import math
    def angle_key(p):
        if p == start:
            return (-float('inf'), 0)
        dx, dy = p[0] - start[0], p[1] - start[1]
        return (math.atan2(dy, dx), dx*dx + dy*dy)

    sorted_points = sorted(points, key=angle_key)
    hull = []
    for p in sorted_points:
        while len(hull) >= 2 and cross(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)
    return hull


def point_in_polygon(point, polygon):
    """点が多角形内にあるか"""
    if len(polygon) < 3:
        return False

    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1

    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside


def analyze_stability(objects):
    """安定性を解析"""
    centroid = calculate_combined_centroid(objects)
    ground_points = find_support_polygon(objects)
    support_polygon = compute_convex_hull_2d(ground_points)
    is_stable = point_in_polygon((centroid.x, centroid.y), support_polygon)
    return is_stable, centroid


def export_obj(objects, filepath):
    """OBJエクスポート"""
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects:
        obj.select_set(True)
    bpy.ops.wm.obj_export(filepath=filepath, export_selected_objects=True)


def generate_single(index, total):
    """1つのサンプルを生成・解析・保存"""
    clear_scene()

    # 家具タイプをランダム選択（50%テーブル、50%椅子）
    furniture_type = random.choice(['table', 'chair'])

    # パラメータ生成（バランスよく安定/不安定を混ぜる）
    bias_choice = random.choice(['balanced', 'stable', 'unstable', 'very_unstable'])

    if furniture_type == 'table':
        params = generate_table_params(bias=bias_choice)
        objects = create_table_with_params(params)
    else:
        params = generate_chair_params(bias=bias_choice)
        objects = create_chair_with_params(params)

    # 安定性解析
    is_stable, centroid = analyze_stability(objects)

    # 出力先決定
    if is_stable:
        output_dir = STABLE_DIR
    else:
        output_dir = UNSTABLE_DIR

    # ファイル名（家具タイプを含める）
    filename = f"{furniture_type}_{index:05d}.obj"
    filepath = os.path.join(output_dir, filename)

    # エクスポート
    export_obj(objects, filepath)

    # メタデータ保存
    meta = {
        "id": index,
        "type": furniture_type,
        "stable": is_stable,
        "params": params,
        "centroid": {"x": centroid.x, "y": centroid.y, "z": centroid.z}
    }
    meta_path = filepath.replace('.obj', '.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    return is_stable, furniture_type


def main():
    args = get_script_args()
    num_samples = int(args[0]) if args else 10

    print("\n" + "=" * 60)
    print(f"家具データセット生成パイプライン開始")
    print(f"生成数: {num_samples} (テーブル + 椅子)")
    print("=" * 60 + "\n")

    setup_directories()

    stable_count = 0
    unstable_count = 0
    table_count = 0
    chair_count = 0

    for i in range(num_samples):
        is_stable, furniture_type = generate_single(i, num_samples)

        if is_stable:
            stable_count += 1
        else:
            unstable_count += 1

        if furniture_type == 'table':
            table_count += 1
        else:
            chair_count += 1

        # 進捗表示（10%ごと）
        if (i + 1) % max(1, num_samples // 10) == 0:
            progress = (i + 1) / num_samples * 100
            print(f"進捗: {i+1}/{num_samples} ({progress:.0f}%) - "
                  f"Stable: {stable_count}, Unstable: {unstable_count} | "
                  f"Table: {table_count}, Chair: {chair_count}")

    print("\n" + "=" * 60)
    print("生成完了!")
    print(f"  Stable:   {stable_count} ({stable_count/num_samples*100:.1f}%)")
    print(f"  Unstable: {unstable_count} ({unstable_count/num_samples*100:.1f}%)")
    print(f"  Tables:   {table_count}")
    print(f"  Chairs:   {chair_count}")
    print(f"\n保存先: {DATASET_DIR}")
    print("=" * 60 + "\n")

    # サマリーJSON
    summary = {
        "total": num_samples,
        "stable": stable_count,
        "unstable": unstable_count,
        "tables": table_count,
        "chairs": chair_count
    }
    with open(os.path.join(DATASET_DIR, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
