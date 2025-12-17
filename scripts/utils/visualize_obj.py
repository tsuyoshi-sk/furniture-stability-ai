"""
OBJファイルを画像としてレンダリング（Blender使用）
GUIなしで3Dモデルを確認するためのツール
"""
import bpy
import sys
import os
import math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_script_args():
    argv = sys.argv
    if "--" in argv:
        return argv[argv.index("--") + 1:]
    return []


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def setup_camera():
    """カメラを設定"""
    bpy.ops.object.camera_add(location=(2.5, -2.5, 2.0))
    camera = bpy.context.active_object
    camera.name = "Camera"

    # カメラを原点に向ける
    direction = mathutils.Vector((0, 0, 0.5)) - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()

    bpy.context.scene.camera = camera
    return camera


def setup_lighting():
    """照明を設定"""
    # メインライト
    bpy.ops.object.light_add(type='SUN', location=(3, -3, 5))
    sun = bpy.context.active_object
    sun.data.energy = 3

    # フィルライト
    bpy.ops.object.light_add(type='AREA', location=(-2, 2, 3))
    fill = bpy.context.active_object
    fill.data.energy = 100
    fill.data.size = 3


def import_and_center(filepath):
    """OBJをインポートして中央に配置"""
    bpy.ops.wm.obj_import(filepath=filepath)

    # 全オブジェクトを選択
    imported = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']

    if not imported:
        return None

    # バウンディングボックスを計算
    min_co = [float('inf')] * 3
    max_co = [float('-inf')] * 3

    for obj in imported:
        for v in obj.data.vertices:
            world_co = obj.matrix_world @ v.co
            for i in range(3):
                min_co[i] = min(min_co[i], world_co[i])
                max_co[i] = max(max_co[i], world_co[i])

    # 中心を計算
    center = [(min_co[i] + max_co[i]) / 2 for i in range(3)]

    # 床面に接地させる（Z方向のみ調整）
    offset_z = -min_co[2]

    for obj in imported:
        obj.location.x -= center[0]
        obj.location.y -= center[1]
        obj.location.z += offset_z

    # マテリアル設定
    mat = bpy.data.materials.new(name="FurnitureMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.6, 0.4, 0.2, 1)  # 木の色
    bsdf.inputs["Roughness"].default_value = 0.5

    for obj in imported:
        obj.data.materials.clear()
        obj.data.materials.append(mat)

    return imported


def setup_render():
    """レンダリング設定"""
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 64

    scene.render.resolution_x = 800
    scene.render.resolution_y = 600
    scene.render.film_transparent = True


def add_floor():
    """床を追加"""
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.name = "Floor"

    mat = bpy.data.materials.new(name="FloorMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.9, 0.9, 0.9, 1)
    bsdf.inputs["Roughness"].default_value = 0.8

    floor.data.materials.append(mat)


def render_image(output_path):
    """画像をレンダリング"""
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    print(f"画像保存: {output_path}")


def main():
    import mathutils
    global mathutils

    args = get_script_args()

    if not args:
        print("使用法: ./run_blender.sh visualize_obj.py -- input.obj [output.png]")
        return

    input_file = args[0]
    if len(args) > 1:
        output_file = args[1]
    else:
        output_file = os.path.splitext(input_file)[0] + "_render.png"

    if not os.path.exists(input_file):
        print(f"エラー: ファイルが見つかりません: {input_file}")
        return

    print(f"\n{'=' * 50}")
    print("OBJ可視化ツール")
    print(f"{'=' * 50}")
    print(f"入力: {input_file}")
    print(f"出力: {output_file}")

    # シーン準備
    clear_scene()

    # インポート
    objects = import_and_center(input_file)
    if not objects:
        print("エラー: オブジェクトをインポートできませんでした")
        return

    # シーン設定
    add_floor()
    setup_lighting()

    # カメラ設定（mathutilsをインポート後に実行）
    bpy.ops.object.camera_add(location=(2.5, -2.5, 2.0))
    camera = bpy.context.active_object
    camera.name = "Camera"
    direction = mathutils.Vector((0, 0, 0.5)) - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    bpy.context.scene.camera = camera

    setup_render()

    # レンダリング
    render_image(output_file)

    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()
