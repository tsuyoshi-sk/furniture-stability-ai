bl_info = {
    "name": "Furniture Stability Analyzer",
    "author": "Furniture Research Project",
    "version": (1, 0, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Furniture",
    "description": "Analyze furniture stability and get design improvement suggestions",
    "category": "Object",
}

import bpy
import subprocess
import json
import tempfile
import os
from pathlib import Path


# =============================================================================
# プロパティ
# =============================================================================

class FurnitureStabilityProperties(bpy.types.PropertyGroup):
    """アドオンのプロパティ"""

    python_path: bpy.props.StringProperty(
        name="Python Path",
        description="Path to Python executable with required packages",
        default="python3",
        subtype='FILE_PATH'
    )

    scripts_path: bpy.props.StringProperty(
        name="Scripts Path",
        description="Path to furniture_research/scripts directory",
        default="",
        subtype='DIR_PATH'
    )

    last_result: bpy.props.StringProperty(
        name="Last Result",
        default=""
    )

    show_details: bpy.props.BoolProperty(
        name="Show Details",
        default=True
    )


# =============================================================================
# オペレーター
# =============================================================================

class FURNITURE_OT_analyze_stability(bpy.types.Operator):
    """選択オブジェクトの安定性を解析"""
    bl_idname = "furniture.analyze_stability"
    bl_label = "Analyze Stability"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.type == 'MESH'

    def execute(self, context):
        props = context.scene.furniture_stability

        # スクリプトパスを確認
        scripts_path = bpy.path.abspath(props.scripts_path)
        if not scripts_path or not os.path.exists(scripts_path):
            self.report({'ERROR'}, "Scripts path not set. Please configure in panel.")
            return {'CANCELLED'}

        obj = context.active_object

        # 一時ファイルにエクスポート
        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # OBJとしてエクスポート
            bpy.ops.wm.obj_export(
                filepath=tmp_path,
                export_selected_objects=True,
                export_materials=False,
                export_uv=False,
                export_normals=False,
                apply_modifiers=True,
                global_scale=1.0,
                forward_axis='NEGATIVE_Z',
                up_axis='Y'
            )

            # 解析スクリプトを実行
            design_advisor_path = os.path.join(scripts_path, "utils", "design_advisor.py")

            result = subprocess.run(
                [props.python_path, design_advisor_path, tmp_path, "-o", tmp_path + ".json"],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(scripts_path)
            )

            # 結果を読み込み
            json_path = tmp_path + ".json"
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    analysis = json.load(f)

                props.last_result = json.dumps(analysis, ensure_ascii=False)

                # カスタムプロパティに保存
                obj["stability"] = analysis.get("current_stability", "unknown")
                obj["confidence"] = analysis.get("confidence", 0)
                obj["stability_score"] = analysis.get("physics", {}).get("stability_score", 0)
                obj["issues"] = json.dumps(analysis.get("issues", []), ensure_ascii=False)
                obj["suggestions"] = json.dumps(
                    [s.get("suggestion", "") for s in analysis.get("suggestions", [])],
                    ensure_ascii=False
                )

                # レポート
                if analysis.get("current_stability") == "stable":
                    self.report({'INFO'}, f"Stable ({analysis.get('confidence', 0)*100:.0f}% confidence)")
                else:
                    issues = analysis.get("issues", [])
                    self.report({'WARNING'}, f"Unstable: {', '.join(issues[:2])}")

                os.unlink(json_path)

            else:
                self.report({'ERROR'}, f"Analysis failed: {result.stderr}")
                return {'CANCELLED'}

        except Exception as e:
            self.report({'ERROR'}, f"Error: {str(e)}")
            return {'CANCELLED'}

        finally:
            # クリーンアップ
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        return {'FINISHED'}


class FURNITURE_OT_analyze_all(bpy.types.Operator):
    """シーン内のすべてのメッシュを解析"""
    bl_idname = "furniture.analyze_all"
    bl_label = "Analyze All Objects"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        mesh_objects = [obj for obj in context.scene.objects if obj.type == 'MESH']

        if not mesh_objects:
            self.report({'WARNING'}, "No mesh objects in scene")
            return {'CANCELLED'}

        analyzed = 0
        unstable = 0

        for obj in mesh_objects:
            # オブジェクトを選択
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            context.view_layer.objects.active = obj

            # 解析実行
            result = bpy.ops.furniture.analyze_stability()
            if result == {'FINISHED'}:
                analyzed += 1
                if obj.get("stability") == "unstable":
                    unstable += 1

        self.report({'INFO'}, f"Analyzed {analyzed} objects, {unstable} unstable")
        return {'FINISHED'}


class FURNITURE_OT_show_suggestions(bpy.types.Operator):
    """改善提案を表示"""
    bl_idname = "furniture.show_suggestions"
    bl_label = "Show Suggestions"

    def execute(self, context):
        props = context.scene.furniture_stability

        if not props.last_result:
            self.report({'WARNING'}, "No analysis result. Run analysis first.")
            return {'CANCELLED'}

        try:
            analysis = json.loads(props.last_result)
            suggestions = analysis.get("suggestions", [])

            if suggestions:
                # ポップアップダイアログで表示
                def draw(self, context):
                    layout = self.layout
                    for i, s in enumerate(suggestions, 1):
                        box = layout.box()
                        box.label(text=f"{i}. {s.get('suggestion', '')}", icon='MODIFIER')
                        box.label(text=f"   {s.get('specific_action', '')}")

                context.window_manager.popup_menu(draw, title="Improvement Suggestions")

        except json.JSONDecodeError:
            self.report({'ERROR'}, "Invalid result data")
            return {'CANCELLED'}

        return {'FINISHED'}


class FURNITURE_OT_highlight_unstable(bpy.types.Operator):
    """不安定なオブジェクトをハイライト"""
    bl_idname = "furniture.highlight_unstable"
    bl_label = "Highlight Unstable"

    def execute(self, context):
        unstable_count = 0

        for obj in context.scene.objects:
            if obj.type == 'MESH' and obj.get("stability") == "unstable":
                obj.select_set(True)
                # 赤色のマテリアルを適用（オプション）
                unstable_count += 1
            else:
                obj.select_set(False)

        self.report({'INFO'}, f"Selected {unstable_count} unstable objects")
        return {'FINISHED'}


# =============================================================================
# パネル
# =============================================================================

class FURNITURE_PT_main_panel(bpy.types.Panel):
    """メインパネル"""
    bl_label = "Furniture Stability"
    bl_idname = "FURNITURE_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Furniture'

    def draw(self, context):
        layout = self.layout
        props = context.scene.furniture_stability
        obj = context.active_object

        # 設定
        box = layout.box()
        box.label(text="Settings", icon='PREFERENCES')
        box.prop(props, "python_path")
        box.prop(props, "scripts_path")

        layout.separator()

        # 解析ボタン
        col = layout.column(align=True)
        col.scale_y = 1.5
        col.operator("furniture.analyze_stability", icon='VIEWZOOM')
        col.operator("furniture.analyze_all", icon='OUTLINER_OB_GROUP_INSTANCE')

        layout.separator()

        # 結果表示
        if obj and obj.type == 'MESH':
            box = layout.box()
            box.label(text=f"Object: {obj.name}", icon='MESH_DATA')

            stability = obj.get("stability")
            if stability:
                row = box.row()
                if stability == "stable":
                    row.label(text="Status: STABLE", icon='CHECKMARK')
                else:
                    row.label(text="Status: UNSTABLE", icon='ERROR')

                confidence = obj.get("confidence", 0)
                box.label(text=f"Confidence: {confidence*100:.1f}%")

                score = obj.get("stability_score", 0)
                box.label(text=f"Score: {score:.2f} / 2.5")

                # 問題点
                issues_str = obj.get("issues", "[]")
                try:
                    issues = json.loads(issues_str)
                    if issues:
                        box.label(text="Issues:", icon='ERROR')
                        for issue in issues[:3]:
                            box.label(text=f"  - {issue}")
                except:
                    pass

                # 提案
                suggestions_str = obj.get("suggestions", "[]")
                try:
                    suggestions = json.loads(suggestions_str)
                    if suggestions:
                        box.label(text="Suggestions:", icon='LIGHT')
                        for s in suggestions[:3]:
                            box.label(text=f"  - {s}")
                except:
                    pass

        layout.separator()

        # ユーティリティ
        row = layout.row(align=True)
        row.operator("furniture.highlight_unstable", icon='RESTRICT_SELECT_OFF')
        row.operator("furniture.show_suggestions", icon='INFO')


# =============================================================================
# 登録
# =============================================================================

classes = (
    FurnitureStabilityProperties,
    FURNITURE_OT_analyze_stability,
    FURNITURE_OT_analyze_all,
    FURNITURE_OT_show_suggestions,
    FURNITURE_OT_highlight_unstable,
    FURNITURE_PT_main_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.furniture_stability = bpy.props.PointerProperty(type=FurnitureStabilityProperties)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.furniture_stability


if __name__ == "__main__":
    register()
