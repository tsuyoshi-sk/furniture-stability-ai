"""
3Dモデルの構造解析スクリプト
- 体積計算
- 重心位置計算
- 支持基底面の計算
- 安定性判定（重心が支持基底面内にあるか）

使用法: ./run_blender.sh analyze_structure.py -- input_file.obj
"""
import bpy
import bmesh
import json
import os
import sys
from mathutils import Vector

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_script_args():
    """Blender経由で渡された引数を取得"""
    argv = sys.argv
    if "--" in argv:
        return argv[argv.index("--") + 1:]
    return []


def clear_scene():
    """シーンをクリア"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def import_obj(filepath):
    """OBJファイルをインポート"""
    bpy.ops.wm.obj_import(filepath=filepath)
    return list(bpy.context.selected_objects)


def calculate_volume_and_centroid(obj):
    """
    オブジェクトの体積と重心を計算
    BMeshを使用して三角形分割後に計算
    """
    # オブジェクトのメッシュデータを取得
    bm = bmesh.new()
    bm.from_mesh(obj.data)

    # ワールド座標に変換
    bm.transform(obj.matrix_world)

    # 三角形分割
    bmesh.ops.triangulate(bm, faces=bm.faces[:])

    total_volume = 0.0
    weighted_centroid = Vector((0.0, 0.0, 0.0))

    # 各三角形について符号付き体積を計算（原点を基準とした四面体）
    for face in bm.faces:
        if len(face.verts) == 3:
            v0 = face.verts[0].co
            v1 = face.verts[1].co
            v2 = face.verts[2].co

            # 四面体の符号付き体積 (原点からの)
            signed_vol = v0.dot(v1.cross(v2)) / 6.0
            total_volume += signed_vol

            # この四面体の重心
            tetra_centroid = (v0 + v1 + v2) / 4.0  # 原点を含む4点の平均
            weighted_centroid += tetra_centroid * signed_vol

    bm.free()

    volume = abs(total_volume)
    if volume > 1e-10:
        centroid = weighted_centroid / total_volume
    else:
        centroid = Vector((0.0, 0.0, 0.0))

    return volume, centroid


def calculate_combined_centroid(objects):
    """
    複数オブジェクトの合成重心を計算
    各オブジェクトの体積で重み付け
    """
    total_volume = 0.0
    weighted_centroid = Vector((0.0, 0.0, 0.0))

    object_data = []

    for obj in objects:
        if obj.type != 'MESH':
            continue

        volume, centroid = calculate_volume_and_centroid(obj)

        object_data.append({
            'name': obj.name,
            'volume': volume,
            'centroid': list(centroid)
        })

        total_volume += volume
        weighted_centroid += centroid * volume

    if total_volume > 1e-10:
        combined_centroid = weighted_centroid / total_volume
    else:
        combined_centroid = Vector((0.0, 0.0, 0.0))

    return total_volume, combined_centroid, object_data


def find_support_polygon(objects):
    """
    支持基底面（接地点から形成される多角形）を計算
    床面（Z=最小値）に接触している頂点を抽出
    """
    ground_points = []
    min_z = float('inf')

    # まず全体の最小Z座標を見つける
    for obj in objects:
        if obj.type != 'MESH':
            continue

        mesh = obj.data
        world_matrix = obj.matrix_world

        for vert in mesh.vertices:
            world_co = world_matrix @ vert.co
            if world_co.z < min_z:
                min_z = world_co.z

    # 最小Zに近い点（許容誤差内）を接地点とする
    tolerance = 0.01  # 1cm

    for obj in objects:
        if obj.type != 'MESH':
            continue

        mesh = obj.data
        world_matrix = obj.matrix_world

        for vert in mesh.vertices:
            world_co = world_matrix @ vert.co
            if abs(world_co.z - min_z) < tolerance:
                ground_points.append((world_co.x, world_co.y))

    return ground_points, min_z


def compute_convex_hull_2d(points):
    """
    2D点群の凸包を計算（Graham scan）
    """
    if len(points) < 3:
        return points

    # 最も下にある点を開始点とする
    points = sorted(set(points))
    start = min(points, key=lambda p: (p[1], p[0]))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # 角度でソート
    import math
    def angle_key(p):
        if p == start:
            return (-float('inf'), 0)
        dx = p[0] - start[0]
        dy = p[1] - start[1]
        return (math.atan2(dy, dx), dx*dx + dy*dy)

    sorted_points = sorted(points, key=angle_key)

    hull = []
    for p in sorted_points:
        while len(hull) >= 2 and cross(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)

    return hull


def point_in_polygon(point, polygon):
    """
    点が多角形の内部にあるか判定（Ray casting法）
    point: (x, y)
    polygon: [(x1, y1), (x2, y2), ...]
    """
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


def calculate_stability_margin(centroid_xy, support_polygon):
    """
    安定性マージンを計算
    重心から支持多角形の最近接エッジまでの距離
    正の値: 安定（内側にある）
    負の値: 不安定（外側にある）
    """
    if len(support_polygon) < 3:
        return -float('inf')

    import math

    def point_to_segment_distance(px, py, x1, y1, x2, y2):
        """点から線分への距離"""
        dx = x2 - x1
        dy = y2 - y1
        length_sq = dx * dx + dy * dy

        if length_sq < 1e-10:
            return math.sqrt((px - x1)**2 + (py - y1)**2)

        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / length_sq))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy

        return math.sqrt((px - proj_x)**2 + (py - proj_y)**2)

    cx, cy = centroid_xy
    min_distance = float('inf')

    n = len(support_polygon)
    for i in range(n):
        x1, y1 = support_polygon[i]
        x2, y2 = support_polygon[(i + 1) % n]

        dist = point_to_segment_distance(cx, cy, x1, y1, x2, y2)
        min_distance = min(min_distance, dist)

    # 内側にあれば正、外側にあれば負
    is_inside = point_in_polygon((cx, cy), support_polygon)

    return min_distance if is_inside else -min_distance


def analyze_structure(input_file):
    """メイン解析関数"""
    print("\n" + "=" * 50)
    print("構造解析開始")
    print("=" * 50)
    print(f"入力ファイル: {input_file}")

    # シーンをクリア
    clear_scene()

    # OBJをインポート
    objects = import_obj(input_file)
    print(f"読み込んだオブジェクト数: {len(objects)}")

    # 体積と重心を計算
    total_volume, centroid, object_data = calculate_combined_centroid(objects)
    print(f"\n合計体積: {total_volume:.6f} m³")
    print(f"合成重心: ({centroid.x:.4f}, {centroid.y:.4f}, {centroid.z:.4f})")

    # 支持基底面を計算
    ground_points, ground_z = find_support_polygon(objects)
    support_polygon = compute_convex_hull_2d(ground_points)
    print(f"\n接地点数: {len(ground_points)}")
    print(f"支持多角形の頂点数: {len(support_polygon)}")
    print(f"床面高さ: {ground_z:.4f}")

    # 安定性判定
    centroid_xy = (centroid.x, centroid.y)
    is_stable = point_in_polygon(centroid_xy, support_polygon)
    stability_margin = calculate_stability_margin(centroid_xy, support_polygon)

    print(f"\n=== 安定性判定 ===")
    print(f"重心のXY座標: ({centroid.x:.4f}, {centroid.y:.4f})")
    print(f"安定性: {'安定 (Stable)' if is_stable else '不安定 (Unstable)'}")
    print(f"安定性マージン: {stability_margin:.4f} m")

    # 結果をJSONで保存
    result = {
        "input_file": os.path.basename(input_file),
        "stable": is_stable,
        "stability_margin": round(stability_margin, 6),
        "center_of_mass": {
            "x": round(centroid.x, 6),
            "y": round(centroid.y, 6),
            "z": round(centroid.z, 6)
        },
        "total_volume": round(total_volume, 6),
        "ground_level": round(ground_z, 6),
        "support_polygon": [{"x": round(p[0], 6), "y": round(p[1], 6)} for p in support_polygon],
        "objects": object_data
    }

    output_file = os.path.join(OUTPUT_DIR, "result.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n結果を保存: {output_file}")
    print("=" * 50)

    return result


def main():
    args = get_script_args()

    if args:
        input_file = args[0]
    else:
        # デフォルトは同じディレクトリのoutput_table.obj
        input_file = os.path.join(OUTPUT_DIR, "output_table.obj")

    if not os.path.exists(input_file):
        print(f"エラー: ファイルが見つかりません: {input_file}")
        return

    analyze_structure(input_file)


if __name__ == "__main__":
    main()
