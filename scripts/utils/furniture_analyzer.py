#!/usr/bin/env python3
"""
Furniture Analyzer - Unified tool for stability and load capacity prediction
- Interactive material selection
- Combined stability + load capacity analysis
- Visual output
"""
import sys
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from inference import FurnitureStabilityPredictor
from load_capacity import (
    LoadCapacityPredictor, MATERIALS, LoadCapacityResult,
    analyze_shelf_geometry, load_obj_vertices, load_obj_faces, format_result,
    calculate_load_range, estimate_material_distribution
)


# =============================================================================
# Material Database with Japanese names
# =============================================================================

MATERIAL_INFO = {
    'pine': {
        'ja': 'パイン材（松）',
        'en': 'Pine Wood',
        'description': '軽量で加工しやすい針葉樹材',
        'typical_use': '棚板、DIY家具',
        'strength': '中',
    },
    'oak': {
        'ja': 'オーク材（楢）',
        'en': 'Oak Wood',
        'description': '硬く耐久性の高い広葉樹材',
        'typical_use': '高級家具、フローリング',
        'strength': '高',
    },
    'plywood': {
        'ja': '合板',
        'en': 'Plywood',
        'description': '薄い板を交互に重ねた積層材',
        'typical_use': '一般的な棚板、家具',
        'strength': '中',
    },
    'mdf': {
        'ja': 'MDF（中密度繊維板）',
        'en': 'MDF',
        'description': '木質繊維を圧縮した板',
        'typical_use': 'カラーボックス、安価な家具',
        'strength': '低〜中',
    },
    'particle_board': {
        'ja': 'パーティクルボード',
        'en': 'Particle Board',
        'description': '木材チップを圧縮した板',
        'typical_use': '安価な家具、芯材',
        'strength': '低',
    },
    'steel': {
        'ja': 'スチール（鋼）',
        'en': 'Steel',
        'description': '高強度の金属材',
        'typical_use': 'メタルラック、業務用棚',
        'strength': '非常に高',
    },
    'aluminum': {
        'ja': 'アルミニウム',
        'en': 'Aluminum',
        'description': '軽量で錆びにくい金属',
        'typical_use': '軽量棚、屋外家具',
        'strength': '高',
    },
    'plastic_abs': {
        'ja': 'ABS樹脂',
        'en': 'ABS Plastic',
        'description': '耐衝撃性のあるプラスチック',
        'typical_use': 'プラスチック家具',
        'strength': '低〜中',
    },
}


def print_material_menu():
    """Print material selection menu"""
    print("\n" + "=" * 60)
    print("材質を選択してください")
    print("=" * 60)

    for i, (key, info) in enumerate(MATERIAL_INFO.items(), 1):
        mat = MATERIALS[key]
        print(f"\n  [{i}] {info['ja']}")
        print(f"      強度: {info['strength']} | E={mat.elastic_modulus}MPa")
        print(f"      用途: {info['typical_use']}")

    print(f"\n  [0] 材質不明（自動推定）")
    print("-" * 60)


def select_material_interactive() -> Optional[str]:
    """Interactive material selection"""
    print_material_menu()

    material_keys = list(MATERIAL_INFO.keys())

    while True:
        try:
            choice = input("\n番号を入力 (0-8): ").strip()
            if choice == '':
                return None

            choice_int = int(choice)
            if choice_int == 0:
                return None
            elif 1 <= choice_int <= len(material_keys):
                selected = material_keys[choice_int - 1]
                info = MATERIAL_INFO[selected]
                print(f"\n→ {info['ja']} を選択しました")
                return selected
            else:
                print("無効な番号です")
        except ValueError:
            print("数字を入力してください")
        except KeyboardInterrupt:
            return None


def analyze_furniture(obj_path: Path, material: Optional[str] = None,
                      interactive: bool = True) -> Dict:
    """
    Analyze furniture for both stability and load capacity

    Args:
        obj_path: Path to OBJ file
        material: Material key or None for auto-estimation
        interactive: Whether to prompt for material selection

    Returns:
        Combined analysis results
    """
    obj_path = Path(obj_path)

    if not obj_path.exists():
        raise FileNotFoundError(f"File not found: {obj_path}")

    # Interactive material selection
    if material is None and interactive:
        material = select_material_interactive()

    results = {
        'file': str(obj_path),
        'filename': obj_path.name,
        'material': material,
        'material_info': MATERIAL_INFO.get(material, {}) if material else {},
    }

    # Stability prediction
    print("\n安定性を解析中...")
    try:
        stability_predictor = FurnitureStabilityPredictor()
        stability_result = stability_predictor.predict(obj_path, use_tta=True)
        results['stability'] = {
            'prediction': stability_result['prediction'],
            'confidence': stability_result['confidence'],
            'prob_stable': stability_result['prob_stable'],
            'furniture_type': stability_result['furniture_type'],
            'physics': stability_result['physics'],
        }
    except Exception as e:
        results['stability'] = {'error': str(e)}

    # Load capacity prediction
    print("耐荷重を解析中...")
    try:
        load_predictor = LoadCapacityPredictor()
        load_result = load_predictor.predict(obj_path, material=material)

        # Calculate range if no material specified
        load_range = None
        if material is None:
            vertices = load_obj_vertices(obj_path)
            faces = load_obj_faces(obj_path)
            geometry = analyze_shelf_geometry(vertices, faces)
            load_range = calculate_load_range(geometry)

        results['load_capacity'] = {
            'max_static_load_kg': load_result.max_static_load,
            'max_dynamic_load_kg': load_result.max_dynamic_load,
            'safety_factor': load_result.safety_factor,
            'max_deflection_mm': load_result.max_deflection,
            'weak_point': load_result.weak_point,
            'confidence': load_result.confidence,
            'geometry': {
                'span_mm': load_result.geometry.span_length,
                'width_mm': load_result.geometry.width,
                'thickness_mm': load_result.geometry.thickness,
                'is_shelf_board': load_result.geometry.is_shelf_board,
            },
            'warnings': load_result.warnings,
            'material_specified': material is not None,
        }

        # Add range info if no material specified
        if load_range:
            results['load_capacity']['range'] = {
                'min_static_kg': load_range['min_static'],
                'max_static_kg': load_range['max_static'],
                'min_dynamic_kg': load_range['min_dynamic'],
                'max_dynamic_kg': load_range['max_dynamic'],
                'uncertainty_ratio': load_range['uncertainty_ratio'],
                'by_material': load_range['by_material'],
            }

    except Exception as e:
        results['load_capacity'] = {'error': str(e)}

    return results


def print_results(results: Dict):
    """Print analysis results"""
    print("\n")
    print("=" * 60)
    print("解析結果")
    print("=" * 60)

    print(f"\nファイル: {results['filename']}")

    if results.get('material'):
        info = results['material_info']
        print(f"材質: {info.get('ja', results['material'])}")
    else:
        print("材質: 自動推定")

    # Stability
    print("\n" + "-" * 60)
    print("【安定性】")
    print("-" * 60)

    stab = results.get('stability', {})
    if 'error' in stab:
        print(f"  エラー: {stab['error']}")
    else:
        pred = stab.get('prediction', 'N/A')
        conf = stab.get('confidence', 0) * 100

        if pred == 'stable':
            status = "✓ 安定"
            color_start = "\033[92m"  # Green
        else:
            status = "✗ 不安定"
            color_start = "\033[91m"  # Red
        color_end = "\033[0m"

        print(f"  判定: {color_start}{status}{color_end}")
        print(f"  信頼度: {conf:.1f}%")
        print(f"  家具タイプ: {stab.get('furniture_type', 'unknown')}")

        physics = stab.get('physics', {})
        if physics:
            print(f"  物理スコア: {physics.get('stability_score', 0):.2f}/2.5")

    # Load Capacity
    print("\n" + "-" * 60)
    print("【耐荷重】")
    print("-" * 60)

    load = results.get('load_capacity', {})
    if 'error' in load:
        print(f"  エラー: {load['error']}")
    else:
        static_load = load.get('max_static_load_kg', 0)
        dynamic_load = load.get('max_dynamic_load_kg', 0)
        confidence = load.get('confidence', 0) * 100
        material_specified = load.get('material_specified', False)

        if material_specified:
            # 材質指定あり - 高精度
            print(f"  静荷重上限: {static_load:.1f} kg")
            print(f"  動荷重上限: {dynamic_load:.1f} kg")
            print(f"  安全率: {load.get('safety_factor', 0)}")
            print(f"  信頼度: \033[92m{confidence:.0f}%\033[0m (材質指定)")
        else:
            # 材質推定 - 範囲表示
            load_range = load.get('range', {})
            if load_range:
                uncertainty = load_range.get('uncertainty_ratio', 1)
                print(f"  静荷重上限: {static_load:.1f} kg (推定)")
                print(f"  動荷重上限: {dynamic_load:.1f} kg (推定)")
                print(f"\n  \033[93m材質不明のため範囲表示:\033[0m")
                print(f"    最小: {load_range.get('min_dynamic_kg', 0):.1f} kg")
                print(f"    最大: {load_range.get('max_dynamic_kg', 0):.1f} kg")
                print(f"    不確実性: {uncertainty:.1f}倍")

                by_material = load_range.get('by_material', {})
                if by_material:
                    print(f"\n  材質別推定:")
                    for mat_key, data in sorted(by_material.items(),
                                                 key=lambda x: -x[1].get('prob', 0)):
                        mat_info = MATERIAL_INFO.get(mat_key, {})
                        mat_name = mat_info.get('ja', mat_key)
                        prob = data.get('prob', 0) * 100
                        dyn = data.get('dynamic', 0)
                        print(f"    {mat_name}: {dyn:.1f} kg ({prob:.0f}%)")
            else:
                print(f"  静荷重上限: {static_load:.1f} kg")
                print(f"  動荷重上限: {dynamic_load:.1f} kg")

            print(f"\n  \033[93m※材質を指定すると精度が大幅に向上します\033[0m")

        print(f"  安全率: {load.get('safety_factor', 0)}")

        geo = load.get('geometry', {})
        if geo:
            print(f"\n  寸法:")
            print(f"    スパン: {geo.get('span_mm', 0):.0f} mm")
            print(f"    幅: {geo.get('width_mm', 0):.0f} mm")
            print(f"    厚さ: {geo.get('thickness_mm', 0):.0f} mm")

        warnings = load.get('warnings', [])
        if warnings:
            print(f"\n  注意:")
            for w in warnings:
                print(f"    - {w}")

    # Recommendation
    print("\n" + "-" * 60)
    print("【推奨事項】")
    print("-" * 60)

    if 'error' not in stab and 'error' not in load:
        if stab.get('prediction') == 'unstable':
            print("  ⚠ 安定性に問題があります。設置方法を見直してください。")

        dynamic_load = load.get('max_dynamic_load_kg', 0)
        if dynamic_load < 5:
            print("  ⚠ 耐荷重が低いです。軽い物のみを載せてください。")
        elif dynamic_load < 20:
            print("  △ 一般的な書籍程度まで対応可能です。")
        elif dynamic_load < 50:
            print("  ○ 一般的な家庭用途に適しています。")
        else:
            print("  ◎ 十分な耐荷重があります。")

    print("\n" + "=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='家具解析ツール（安定性 + 耐荷重）')
    parser.add_argument('input', nargs='?', help='OBJファイル')
    parser.add_argument('--material', '-m', choices=list(MATERIALS.keys()),
                        help='材質（指定しない場合は対話選択）')
    parser.add_argument('--no-interactive', '-n', action='store_true',
                        help='対話モード無効（材質自動推定）')
    parser.add_argument('--output', '-o', help='結果をJSONに保存')
    parser.add_argument('--list-materials', action='store_true',
                        help='材質リストを表示')
    args = parser.parse_args()

    if args.list_materials:
        print_material_menu()
        return

    if not args.input:
        print("\n家具解析ツール")
        print("=" * 40)
        print("\n使用例:")
        print("  python3 furniture_analyzer.py shelf.obj")
        print("  python3 furniture_analyzer.py shelf.obj --material plywood")
        print("  python3 furniture_analyzer.py --list-materials")
        return

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path

    if not input_path.exists():
        print(f"エラー: {input_path} が見つかりません")
        return

    interactive = not args.no_interactive and args.material is None

    try:
        results = analyze_furniture(
            input_path,
            material=args.material,
            interactive=interactive
        )
        print_results(results)

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n結果を保存: {args.output}")

    except Exception as e:
        print(f"エラー: {e}")


if __name__ == "__main__":
    main()
