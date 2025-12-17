"""
ランダムな寸法のテーブルを生成し、OBJファイルとして保存するスクリプト
Blender --background モードで実行
"""
import bpy
import random
import os

# 出力ディレクトリ
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def clear_scene():
    """シーン内の全オブジェクトを削除"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def create_table():
    """ランダムな寸法のテーブルを生成"""

    # ランダムパラメータ設定
    table_width = random.uniform(0.8, 1.5)    # 天板の幅 (X軸)
    table_depth = random.uniform(0.5, 1.0)    # 天板の奥行き (Y軸)
    table_height = random.uniform(0.6, 0.9)   # テーブルの高さ
    top_thickness = random.uniform(0.03, 0.06) # 天板の厚さ
    leg_thickness = random.uniform(0.04, 0.08) # 脚の太さ

    print(f"=== テーブルパラメータ ===")
    print(f"天板サイズ: {table_width:.3f} x {table_depth:.3f} x {top_thickness:.3f}")
    print(f"テーブル高さ: {table_height:.3f}")
    print(f"脚の太さ: {leg_thickness:.3f}")

    objects = []

    # 天板を作成
    bpy.ops.mesh.primitive_cube_add(
        size=1,
        location=(0, 0, table_height - top_thickness / 2)
    )
    top = bpy.context.active_object
    top.name = "TableTop"
    top.scale = (table_width / 2, table_depth / 2, top_thickness / 2)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    objects.append(top)

    # 4本の脚を作成
    leg_height = table_height - top_thickness
    leg_positions = [
        (table_width / 2 - leg_thickness, table_depth / 2 - leg_thickness),   # 右奥
        (-table_width / 2 + leg_thickness, table_depth / 2 - leg_thickness),  # 左奥
        (table_width / 2 - leg_thickness, -table_depth / 2 + leg_thickness),  # 右手前
        (-table_width / 2 + leg_thickness, -table_depth / 2 + leg_thickness), # 左手前
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

    return objects, {
        "width": table_width,
        "depth": table_depth,
        "height": table_height,
        "top_thickness": top_thickness,
        "leg_thickness": leg_thickness
    }

def export_obj(objects, filename):
    """オブジェクトをOBJファイルとしてエクスポート"""
    # 全オブジェクトを選択
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects:
        obj.select_set(True)

    # OBJエクスポート
    filepath = os.path.join(OUTPUT_DIR, filename)
    bpy.ops.wm.obj_export(
        filepath=filepath,
        export_selected_objects=True
    )
    print(f"エクスポート完了: {filepath}")
    return filepath

def main():
    print("\n" + "="*50)
    print("テーブル自動生成スクリプト開始")
    print("="*50 + "\n")

    # シーンをクリア
    clear_scene()

    # テーブル生成
    objects, params = create_table()

    # OBJエクスポート
    export_obj(objects, "output_table.obj")

    print("\n" + "="*50)
    print("生成完了!")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
