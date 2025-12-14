"""
ランダムな寸法の椅子を生成し、OBJファイルとして保存するスクリプト
Blender --background モードで実行

構造:
- 座面 (Seat)
- 背もたれ (Backrest)
- 4本の脚 (Legs)
"""
import bpy
import random
import os
import math

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def clear_scene():
    """シーン内の全オブジェクトを削除"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def create_chair():
    """ランダムな寸法の椅子を生成"""

    # ランダムパラメータ設定
    seat_width = random.uniform(0.40, 0.50)      # 座面の幅 (X軸)
    seat_depth = random.uniform(0.40, 0.50)      # 座面の奥行き (Y軸)
    seat_thickness = random.uniform(0.03, 0.05)  # 座面の厚さ
    seat_height = random.uniform(0.40, 0.50)     # 座面の高さ（床から）

    leg_thickness = random.uniform(0.03, 0.05)   # 脚の太さ

    backrest_height = random.uniform(0.35, 0.50) # 背もたれの高さ
    backrest_thickness = random.uniform(0.02, 0.04)  # 背もたれの厚さ
    backrest_angle = random.uniform(0, 15)       # 背もたれの傾斜角度（度）

    print(f"=== 椅子パラメータ ===")
    print(f"座面サイズ: {seat_width:.3f} x {seat_depth:.3f} x {seat_thickness:.3f}")
    print(f"座面高さ: {seat_height:.3f}")
    print(f"脚の太さ: {leg_thickness:.3f}")
    print(f"背もたれ高さ: {backrest_height:.3f}")
    print(f"背もたれ角度: {backrest_angle:.1f}°")

    objects = []

    # 座面を作成
    bpy.ops.mesh.primitive_cube_add(
        size=1,
        location=(0, 0, seat_height + seat_thickness / 2)
    )
    seat = bpy.context.active_object
    seat.name = "Seat"
    seat.scale = (seat_width / 2, seat_depth / 2, seat_thickness / 2)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    objects.append(seat)

    # 4本の脚を作成
    leg_height = seat_height
    leg_inset = leg_thickness  # 脚を少し内側に

    leg_positions = [
        (seat_width / 2 - leg_inset, seat_depth / 2 - leg_inset),    # 右奥
        (-seat_width / 2 + leg_inset, seat_depth / 2 - leg_inset),   # 左奥
        (seat_width / 2 - leg_inset, -seat_depth / 2 + leg_inset),   # 右手前
        (-seat_width / 2 + leg_inset, -seat_depth / 2 + leg_inset),  # 左手前
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
    # 背もたれは座面の後ろ（Y軸正方向）に配置し、少し傾斜させる
    backrest_angle_rad = math.radians(backrest_angle)

    # 背もたれの中心位置を計算
    backrest_center_y = seat_depth / 2 - backrest_thickness / 2
    backrest_center_z = seat_height + seat_thickness + backrest_height / 2

    # 傾斜を考慮した位置調整
    backrest_offset_y = (backrest_height / 2) * math.sin(backrest_angle_rad)
    backrest_offset_z = 0

    bpy.ops.mesh.primitive_cube_add(
        size=1,
        location=(0, backrest_center_y + backrest_offset_y, backrest_center_z)
    )
    backrest = bpy.context.active_object
    backrest.name = "Backrest"
    backrest.scale = (seat_width / 2, backrest_thickness / 2, backrest_height / 2)

    # 背もたれを傾ける（X軸周りに回転）
    backrest.rotation_euler[0] = backrest_angle_rad

    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    objects.append(backrest)

    params = {
        "seat_width": seat_width,
        "seat_depth": seat_depth,
        "seat_thickness": seat_thickness,
        "seat_height": seat_height,
        "leg_thickness": leg_thickness,
        "backrest_height": backrest_height,
        "backrest_thickness": backrest_thickness,
        "backrest_angle": backrest_angle
    }

    return objects, params


def export_obj(objects, filename):
    """オブジェクトをOBJファイルとしてエクスポート"""
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects:
        obj.select_set(True)

    filepath = os.path.join(OUTPUT_DIR, filename)
    bpy.ops.wm.obj_export(
        filepath=filepath,
        export_selected_objects=True
    )
    print(f"エクスポート完了: {filepath}")
    return filepath


def main():
    print("\n" + "=" * 50)
    print("椅子自動生成スクリプト開始")
    print("=" * 50 + "\n")

    # シーンをクリア
    clear_scene()

    # 椅子生成
    objects, params = create_chair()

    # OBJエクスポート
    export_obj(objects, "output_chair.obj")

    print("\n" + "=" * 50)
    print("生成完了!")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
