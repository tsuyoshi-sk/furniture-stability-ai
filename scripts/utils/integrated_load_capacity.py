#!/usr/bin/env python3
"""
Integrated Load Capacity System
Combines shelf material strength + bracket load capacity

Final capacity = min(shelf capacity, bracket capacity)
"""
import sys
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
import json

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from load_capacity import (
    LoadCapacityPredictor, LoadCapacityResult, MATERIALS, Material,
    load_obj_vertices, load_obj_faces, analyze_shelf_geometry,
    calculate_load_capacity, estimate_material_distribution, ShelfGeometry
)
from bracket_database import (
    get_bracket_by_code, get_all_brackets, get_support_by_code,
    calculate_bracket_load_capacity, Bracket, Support, SUPPORTS
)


# =============================================================================
# Integrated Result
# =============================================================================

@dataclass
class IntegratedLoadResult:
    """Combined shelf + bracket load capacity result"""
    # Final combined capacity
    final_static_load_kg: float
    final_dynamic_load_kg: float
    limiting_factor: str  # 'shelf', 'bracket', or 'both'

    # Shelf capacity
    shelf_static_load_kg: float
    shelf_dynamic_load_kg: float
    shelf_material: str
    shelf_geometry: ShelfGeometry

    # Bracket capacity
    bracket_static_load_kg: float
    bracket_dynamic_load_kg: float
    bracket_code: str
    bracket_name: str
    support_code: str
    num_brackets: int

    # Metadata
    safety_factor: float
    confidence: float
    warnings: List[str]
    recommendations: List[str]


# =============================================================================
# Bracket Selection Logic
# =============================================================================

def recommend_bracket_for_shelf(
    geometry: ShelfGeometry,
    material: Optional[str] = None,
    shelf_type: str = 'wood'
) -> List[Dict]:
    """
    Recommend suitable brackets based on shelf geometry

    Args:
        geometry: Shelf geometry
        material: Shelf material (if known)
        shelf_type: 'wood' or 'glass'

    Returns:
        List of recommended brackets with suitability score
    """
    width = geometry.width  # depth of shelf (bracket size should match)
    span = geometry.span_length

    recommendations = []

    for bracket in get_all_brackets():
        if bracket.shelf_type not in [shelf_type, 'both']:
            continue

        # Check if bracket sizes are appropriate for shelf width
        suitable_sizes = [s for s in bracket.sizes if abs(s - width) < 100]
        if not suitable_sizes:
            continue

        best_size = min(suitable_sizes, key=lambda s: abs(s - width))

        # Calculate suitability score
        score = 0.0

        # Size match (closer is better)
        size_diff = abs(best_size - width)
        if size_diff < 30:
            score += 0.4
        elif size_diff < 50:
            score += 0.2

        # Load capacity adequacy
        avg_load = sum(bracket.load_capacity_per_pair.values()) / len(bracket.load_capacity_per_pair)
        if avg_load > 30:
            score += 0.3
        elif avg_load > 20:
            score += 0.2
        elif avg_load > 10:
            score += 0.1

        # Material match
        if bracket.shelf_type == shelf_type:
            score += 0.2

        # Category bonus
        if bracket.category == shelf_type:
            score += 0.1

        recommendations.append({
            'bracket': bracket,
            'best_size_mm': best_size,
            'score': score,
            'avg_load_kg': avg_load,
        })

    # Sort by score
    recommendations.sort(key=lambda x: -x['score'])

    return recommendations[:5]  # Top 5


def estimate_num_brackets(span_length_mm: float) -> int:
    """
    Estimate number of bracket pairs needed based on span length

    Standard rules:
    - Up to 600mm: 2 brackets (1 pair)
    - 600-1200mm: 4 brackets (2 pairs)
    - 1200-1800mm: 6 brackets (3 pairs)
    - 1800mm+: Add pair per 600mm
    """
    if span_length_mm <= 600:
        return 2
    elif span_length_mm <= 1200:
        return 4
    elif span_length_mm <= 1800:
        return 6
    else:
        return 2 * (int(span_length_mm / 600) + 1)


# =============================================================================
# Integrated Calculation
# =============================================================================

def calculate_integrated_load_capacity(
    obj_path: Optional[Path] = None,
    geometry: Optional[ShelfGeometry] = None,
    material: Optional[str] = None,
    bracket_code: Optional[str] = None,
    support_code: str = "ASF-1",
    num_brackets: Optional[int] = None,
    dynamic_factor: float = 2.0,
    safety_factor: float = 2.5
) -> IntegratedLoadResult:
    """
    Calculate combined load capacity for shelf + brackets

    Args:
        obj_path: Path to OBJ file (required if geometry not provided)
        geometry: Pre-calculated geometry (optional)
        material: Material key (e.g., 'plywood', 'oak')
        bracket_code: Bracket product code (e.g., 'A-32')
        support_code: Support code (e.g., 'ASF-1', 'S1B-50/50')
        num_brackets: Number of brackets (auto-calculated if not specified)
        dynamic_factor: Dynamic load factor
        safety_factor: Safety factor

    Returns:
        IntegratedLoadResult with combined analysis
    """
    warnings = []
    recommendations = []

    # Get geometry
    if geometry is None:
        if obj_path is None:
            raise ValueError("Either obj_path or geometry must be provided")
        obj_path = Path(obj_path)
        vertices = load_obj_vertices(obj_path)
        faces = load_obj_faces(obj_path)
        geometry = analyze_shelf_geometry(vertices, faces)

    # Estimate num_brackets if not specified
    if num_brackets is None:
        num_brackets = estimate_num_brackets(geometry.span_length)
        recommendations.append(f"スパン {geometry.span_length:.0f}mm に対して{num_brackets}個のブラケット推奨")

    # Calculate shelf material capacity
    if material and material in MATERIALS:
        mat = MATERIALS[material]
        mat_name = mat.name
    else:
        # Use weighted average from probability distribution
        mat_probs = estimate_material_distribution(geometry)
        mat_name = mat_probs[0][0] if mat_probs else 'plywood'
        mat = MATERIALS[mat_name]
        warnings.append(f"材質不明のため{mat_name}と推定（精度が下がる可能性）")

    shelf_result = calculate_load_capacity(
        geometry=geometry,
        material=mat,
        dynamic_factor=dynamic_factor,
        safety_factor=safety_factor
    )

    shelf_static = shelf_result.max_static_load
    shelf_dynamic = shelf_result.max_dynamic_load

    # Calculate bracket capacity
    bracket = None
    bracket_name = "不明"
    bracket_static = float('inf')
    bracket_dynamic = float('inf')

    if bracket_code:
        bracket = get_bracket_by_code(bracket_code)
        if bracket:
            bracket_name = bracket.name
            # Calculate per-pair capacity
            bracket_load_per_pair = calculate_bracket_load_capacity(
                bracket_code, support_code, 2
            )
            # Scale by number of brackets
            num_pairs = num_brackets / 2
            bracket_static = bracket_load_per_pair * num_pairs
            bracket_dynamic = bracket_static / dynamic_factor
        else:
            warnings.append(f"ブラケット '{bracket_code}' が見つかりません")
    else:
        # Auto-recommend bracket
        rec = recommend_bracket_for_shelf(geometry, material, 'wood')
        if rec:
            top_rec = rec[0]
            recommendations.append(
                f"推奨ブラケット: {top_rec['bracket'].code} ({top_rec['bracket'].name}) - "
                f"サイズ {top_rec['best_size_mm']}mm, 耐荷重 ~{top_rec['avg_load_kg']:.0f}kg/pair"
            )
            # Use recommended bracket for calculation
            bracket = top_rec['bracket']
            bracket_code = bracket.code
            bracket_name = bracket.name
            bracket_load_per_pair = list(bracket.load_capacity_per_pair.values())[0]
            num_pairs = num_brackets / 2
            bracket_static = bracket_load_per_pair * num_pairs
            bracket_dynamic = bracket_static / dynamic_factor
        else:
            warnings.append("適切なブラケットが見つかりません - 手動選択が必要")

    # Determine limiting factor
    if bracket_static < shelf_static * 0.95:
        limiting_factor = "bracket"
        final_static = bracket_static
        final_dynamic = bracket_dynamic
        recommendations.append("金具が耐荷重の制限要因です - より強い金具を検討してください")
    elif shelf_static < bracket_static * 0.95:
        limiting_factor = "shelf"
        final_static = shelf_static
        final_dynamic = shelf_dynamic
        recommendations.append("棚板材質が耐荷重の制限要因です - より強い材質を検討してください")
    else:
        limiting_factor = "both"
        final_static = min(shelf_static, bracket_static)
        final_dynamic = min(shelf_dynamic, bracket_dynamic)

    # Calculate confidence
    confidence = 0.9
    if bracket is None:
        confidence *= 0.7
    if material is None:
        confidence *= 0.7
    if not geometry.is_shelf_board:
        confidence *= 0.8

    return IntegratedLoadResult(
        final_static_load_kg=final_static,
        final_dynamic_load_kg=final_dynamic,
        limiting_factor=limiting_factor,
        shelf_static_load_kg=shelf_static,
        shelf_dynamic_load_kg=shelf_dynamic,
        shelf_material=mat_name,
        shelf_geometry=geometry,
        bracket_static_load_kg=bracket_static if bracket_static != float('inf') else 0,
        bracket_dynamic_load_kg=bracket_dynamic if bracket_dynamic != float('inf') else 0,
        bracket_code=bracket_code or "未指定",
        bracket_name=bracket_name,
        support_code=support_code,
        num_brackets=num_brackets,
        safety_factor=safety_factor,
        confidence=confidence,
        warnings=warnings,
        recommendations=recommendations
    )


# =============================================================================
# Quick Calculation Functions
# =============================================================================

def quick_shelf_capacity(
    span_mm: float,
    width_mm: float,
    thickness_mm: float,
    material: str = 'plywood',
    bracket_code: str = 'A-32',
    support_code: str = 'ASF-1',
    num_brackets: int = 2
) -> Dict:
    """
    Quick calculation without OBJ file

    Args:
        span_mm: Shelf span (length)
        width_mm: Shelf width (depth)
        thickness_mm: Shelf thickness
        material: Material key
        bracket_code: Bracket code
        support_code: Support code
        num_brackets: Number of brackets

    Returns:
        Dict with capacity info
    """
    # Create synthetic geometry
    geometry = ShelfGeometry(
        span_length=span_mm,
        width=width_mm,
        thickness=thickness_mm,
        support_type='simply_supported',
        support_positions=[0, 1],
        cross_section_area=width_mm * thickness_mm,
        moment_of_inertia=(width_mm * thickness_mm**3) / 12,
        section_modulus=(width_mm * thickness_mm**2) / 6,
        num_supports=2,
        is_shelf_board=True
    )

    result = calculate_integrated_load_capacity(
        geometry=geometry,
        material=material,
        bracket_code=bracket_code,
        support_code=support_code,
        num_brackets=num_brackets
    )

    return {
        'final_dynamic_kg': result.final_dynamic_load_kg,
        'final_static_kg': result.final_static_load_kg,
        'limiting_factor': result.limiting_factor,
        'shelf_capacity_kg': result.shelf_dynamic_load_kg,
        'bracket_capacity_kg': result.bracket_dynamic_load_kg,
        'warnings': result.warnings,
        'recommendations': result.recommendations,
    }


# =============================================================================
# Display Functions
# =============================================================================

def format_integrated_result(result: IntegratedLoadResult) -> str:
    """Format integrated result for display"""
    lines = [
        "=" * 70,
        "統合耐荷重解析結果",
        "=" * 70,
        "",
        "【棚板】",
        f"  スパン: {result.shelf_geometry.span_length:.0f} mm",
        f"  幅: {result.shelf_geometry.width:.0f} mm",
        f"  厚さ: {result.shelf_geometry.thickness:.0f} mm",
        f"  材質: {result.shelf_material}",
        f"  棚板耐荷重: {result.shelf_dynamic_load_kg:.1f} kg (動荷重)",
        "",
        "【金具】",
        f"  ブラケット: {result.bracket_code} ({result.bracket_name})",
        f"  支柱: {result.support_code}",
        f"  本数: {result.num_brackets}本",
        f"  金具耐荷重: {result.bracket_dynamic_load_kg:.1f} kg (動荷重)",
        "",
        "-" * 70,
        "【最終耐荷重】",
        "-" * 70,
    ]

    # Color coding based on limiting factor
    if result.limiting_factor == 'bracket':
        factor_text = "金具"
        color = "\033[93m"  # Yellow
    elif result.limiting_factor == 'shelf':
        factor_text = "棚板"
        color = "\033[94m"  # Blue
    else:
        factor_text = "両方同等"
        color = "\033[92m"  # Green

    lines.extend([
        f"  静荷重上限: {result.final_static_load_kg:.1f} kg",
        f"  動荷重上限: {color}{result.final_dynamic_load_kg:.1f} kg\033[0m",
        f"  制限要因: {factor_text}",
        f"  信頼度: {result.confidence*100:.0f}%",
    ])

    if result.warnings:
        lines.append("")
        lines.append("【注意】")
        for w in result.warnings:
            lines.append(f"  ⚠ {w}")

    if result.recommendations:
        lines.append("")
        lines.append("【推奨事項】")
        for r in result.recommendations:
            lines.append(f"  → {r}")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def print_bracket_menu():
    """Print bracket selection menu"""
    print("\n" + "=" * 70)
    print("ブラケット選択")
    print("=" * 70)

    print("\n[木棚用ブラケット]")
    for i, b in enumerate(get_all_brackets()):
        if b.category == 'wood':
            avg_load = sum(b.load_capacity_per_pair.values()) / len(b.load_capacity_per_pair)
            sizes_str = f"{min(b.sizes)}-{max(b.sizes)}mm"
            print(f"  {b.code:10s}: {b.name:20s} | ~{avg_load:.0f}kg/pair | {sizes_str}")

    print("\n[ガラス棚用ブラケット]")
    for b in get_all_brackets():
        if b.category == 'glass':
            avg_load = sum(b.load_capacity_per_pair.values()) / len(b.load_capacity_per_pair)
            sizes_str = f"{min(b.sizes)}-{max(b.sizes)}mm"
            print(f"  {b.code:10s}: {b.name:20s} | ~{avg_load:.0f}kg/pair | {sizes_str}")

    print("\n[支柱タイプ]")
    for s in SUPPORTS:
        print(f"  {s.code:15s}: {s.name}")


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='統合耐荷重計算システム')
    parser.add_argument('input', nargs='?', help='OBJファイル')
    parser.add_argument('--material', '-m', choices=list(MATERIALS.keys()),
                        help='材質')
    parser.add_argument('--bracket', '-b', help='ブラケットコード (例: A-32)')
    parser.add_argument('--support', '-s', default='ASF-1',
                        help='支柱コード (デフォルト: ASF-1)')
    parser.add_argument('--num-brackets', '-n', type=int,
                        help='ブラケット本数')
    parser.add_argument('--list-brackets', action='store_true',
                        help='ブラケット一覧を表示')
    parser.add_argument('--quick', action='store_true',
                        help='クイック計算モード（寸法指定）')
    parser.add_argument('--span', type=float, help='スパン(mm)')
    parser.add_argument('--width', type=float, help='幅(mm)')
    parser.add_argument('--thickness', type=float, help='厚さ(mm)')
    parser.add_argument('--output', '-o', help='結果をJSONに保存')
    args = parser.parse_args()

    if args.list_brackets:
        print_bracket_menu()
        return

    if args.quick:
        if not all([args.span, args.width, args.thickness]):
            print("クイック計算には --span, --width, --thickness が必要です")
            return

        result = quick_shelf_capacity(
            span_mm=args.span,
            width_mm=args.width,
            thickness_mm=args.thickness,
            material=args.material or 'plywood',
            bracket_code=args.bracket or 'A-32',
            support_code=args.support,
            num_brackets=args.num_brackets or 2
        )

        print("\n" + "=" * 50)
        print("クイック計算結果")
        print("=" * 50)
        print(f"\n寸法: {args.span}×{args.width}×{args.thickness} mm")
        print(f"材質: {args.material or 'plywood'}")
        print(f"ブラケット: {args.bracket or 'A-32'}")
        print(f"\n最終耐荷重: {result['final_dynamic_kg']:.1f} kg (動荷重)")
        print(f"制限要因: {result['limiting_factor']}")

        if result['warnings']:
            print("\n注意:")
            for w in result['warnings']:
                print(f"  - {w}")

        if result['recommendations']:
            print("\n推奨:")
            for r in result['recommendations']:
                print(f"  - {r}")
        return

    if not args.input:
        print("\n統合耐荷重計算システム")
        print("=" * 50)
        print("\n使用例:")
        print("  python3 integrated_load_capacity.py shelf.obj")
        print("  python3 integrated_load_capacity.py shelf.obj -m plywood -b A-32")
        print("  python3 integrated_load_capacity.py --quick --span 600 --width 300 --thickness 18")
        print("  python3 integrated_load_capacity.py --list-brackets")
        return

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path

    if not input_path.exists():
        print(f"エラー: {input_path} が見つかりません")
        return

    result = calculate_integrated_load_capacity(
        obj_path=input_path,
        material=args.material,
        bracket_code=args.bracket,
        support_code=args.support,
        num_brackets=args.num_brackets
    )

    print(f"\nファイル: {input_path.name}")
    print(format_integrated_result(result))

    if args.output:
        output_data = {
            'file': str(input_path),
            'final_static_load_kg': result.final_static_load_kg,
            'final_dynamic_load_kg': result.final_dynamic_load_kg,
            'limiting_factor': result.limiting_factor,
            'shelf': {
                'material': result.shelf_material,
                'static_load_kg': result.shelf_static_load_kg,
                'dynamic_load_kg': result.shelf_dynamic_load_kg,
                'geometry': {
                    'span_mm': result.shelf_geometry.span_length,
                    'width_mm': result.shelf_geometry.width,
                    'thickness_mm': result.shelf_geometry.thickness,
                }
            },
            'bracket': {
                'code': result.bracket_code,
                'name': result.bracket_name,
                'support_code': result.support_code,
                'num_brackets': result.num_brackets,
                'static_load_kg': result.bracket_static_load_kg,
                'dynamic_load_kg': result.bracket_dynamic_load_kg,
            },
            'confidence': result.confidence,
            'warnings': result.warnings,
            'recommendations': result.recommendations,
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n結果を保存: {args.output}")


if __name__ == "__main__":
    main()
