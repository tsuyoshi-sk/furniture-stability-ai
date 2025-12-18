#!/usr/bin/env python3
"""
Design Advisor - 家具設計改善提案システム

不安定な家具に対して具体的な改善提案を生成する
"""
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "inference"))

from inference import FurnitureStabilityPredictor, load_obj_vertices, compute_physics_features


# =============================================================================
# 改善提案データ構造
# =============================================================================

@dataclass
class ImprovementSuggestion:
    """改善提案"""
    category: str           # 'cog', 'base', 'aspect', 'balance'
    priority: int           # 1=高, 2=中, 3=低
    issue: str              # 問題点
    suggestion: str         # 提案内容
    expected_improvement: float  # 改善見込み（0-1）
    specific_action: str    # 具体的なアクション
    estimated_values: Dict  # 具体的な数値

@dataclass
class DesignAnalysis:
    """設計分析結果"""
    file_path: str
    furniture_type: str
    current_stability: str      # 'stable' or 'unstable'
    confidence: float
    physics: Dict
    issues: List[str]
    suggestions: List[ImprovementSuggestion]
    estimated_improvement: float
    summary: str


# =============================================================================
# 問題分析
# =============================================================================

def analyze_issues(physics: Dict, prediction: str) -> List[Tuple[str, str, float]]:
    """
    物理特徴から問題点を分析

    Returns:
        List of (category, issue_description, severity)
    """
    issues = []

    cog_height = physics.get('relative_cog_height', 0.5)
    base_area = physics.get('base_area', 0)
    aspect_ratio = physics.get('aspect_ratio', 1.0)
    cog_over_base = physics.get('cog_over_base', True)
    stability_score = physics.get('stability_score', 0)

    # 重心位置の問題
    if cog_height > 0.55:
        severity = min((cog_height - 0.5) * 2, 1.0)
        if cog_height > 0.7:
            issues.append(('cog', '重心位置が非常に高い', severity))
        else:
            issues.append(('cog', '重心位置がやや高い', severity))

    # 底面積の問題
    if base_area < 0.1:
        issues.append(('base', '底面積が非常に小さい', 0.9))
    elif base_area < 0.3:
        issues.append(('base', '底面積が小さい', 0.6))

    # アスペクト比の問題
    if aspect_ratio > 3.0:
        issues.append(('aspect', '非常に縦長な形状（転倒リスク高）', 0.9))
    elif aspect_ratio > 2.0:
        issues.append(('aspect', '縦長な形状', 0.5))

    # 重心位置の問題
    if not cog_over_base:
        issues.append(('balance', '重心が支持底面の外にある', 1.0))

    # 総合スコアが低い
    if stability_score < 1.0 and prediction == 'unstable':
        if not issues:
            issues.append(('general', '総合的な安定性が低い', 0.7))

    return issues


# =============================================================================
# 改善提案生成
# =============================================================================

def generate_suggestions(
    physics: Dict,
    issues: List[Tuple[str, str, float]],
    vertices: np.ndarray
) -> List[ImprovementSuggestion]:
    """問題点に対する改善提案を生成"""

    suggestions = []

    # 寸法を計算
    dims = {
        'width': float(np.max(vertices[:, 0]) - np.min(vertices[:, 0])),
        'height': float(np.max(vertices[:, 1]) - np.min(vertices[:, 1])),
        'depth': float(np.max(vertices[:, 2]) - np.min(vertices[:, 2])),
    }

    cog_height = physics.get('relative_cog_height', 0.5)
    aspect_ratio = physics.get('aspect_ratio', 1.0)

    for category, issue, severity in issues:

        if category == 'cog':
            # 重心を下げる提案
            target_cog = 0.45
            current_cog_mm = cog_height * dims['height']
            target_cog_mm = target_cog * dims['height']
            reduction_mm = current_cog_mm - target_cog_mm

            suggestions.append(ImprovementSuggestion(
                category='cog',
                priority=1 if severity > 0.7 else 2,
                issue=issue,
                suggestion='重心位置を下げる',
                expected_improvement=severity * 0.8,
                specific_action=f'上部を軽量化するか、下部に重量を追加',
                estimated_values={
                    'current_cog_height_ratio': round(cog_height, 2),
                    'target_cog_height_ratio': target_cog,
                    'reduction_mm': round(reduction_mm, 1),
                    'method': '天板の厚みを減らす / 脚を太くする / 下部に棚板を追加'
                }
            ))

            # 高さを下げる提案
            if dims['height'] > 500:
                target_height = dims['height'] * 0.85
                suggestions.append(ImprovementSuggestion(
                    category='cog',
                    priority=2,
                    issue=issue,
                    suggestion='全体の高さを下げる',
                    expected_improvement=severity * 0.6,
                    specific_action=f'高さを{dims["height"] - target_height:.0f}mm短くする',
                    estimated_values={
                        'current_height_mm': round(dims['height'], 1),
                        'target_height_mm': round(target_height, 1),
                        'reduction_mm': round(dims['height'] - target_height, 1)
                    }
                ))

        elif category == 'base':
            # 底面積を広げる提案
            current_base = min(dims['width'], dims['depth'])
            target_base = current_base * 1.3

            suggestions.append(ImprovementSuggestion(
                category='base',
                priority=1 if severity > 0.7 else 2,
                issue=issue,
                suggestion='脚の間隔を広げる',
                expected_improvement=severity * 0.9,
                specific_action=f'脚間隔を{target_base - current_base:.0f}mm広げる',
                estimated_values={
                    'current_base_mm': round(current_base, 1),
                    'target_base_mm': round(target_base, 1),
                    'expansion_mm': round(target_base - current_base, 1),
                    'method': '脚の取り付け位置を外側に / アウトリガーを追加'
                }
            ))

            # ベースプレート追加の提案
            suggestions.append(ImprovementSuggestion(
                category='base',
                priority=2,
                issue=issue,
                suggestion='ベースプレートを追加',
                expected_improvement=severity * 0.7,
                specific_action='底面にベースプレートを取り付ける',
                estimated_values={
                    'recommended_size_mm': f'{dims["width"] * 1.2:.0f} x {dims["depth"] * 1.2:.0f}',
                    'method': '木製または金属製のベースプレートを脚の下に追加'
                }
            ))

        elif category == 'aspect':
            # アスペクト比改善
            target_ratio = 1.8
            if aspect_ratio > target_ratio:
                new_height = dims['height'] * (target_ratio / aspect_ratio)
                new_width = dims['width'] * (aspect_ratio / target_ratio)

                suggestions.append(ImprovementSuggestion(
                    category='aspect',
                    priority=1 if severity > 0.7 else 2,
                    issue=issue,
                    suggestion='高さを低く、または幅を広くする',
                    expected_improvement=severity * 0.85,
                    specific_action='形状のプロポーションを改善',
                    estimated_values={
                        'current_aspect_ratio': round(aspect_ratio, 2),
                        'target_aspect_ratio': target_ratio,
                        'option_a': f'高さを{dims["height"] - new_height:.0f}mm下げる',
                        'option_b': f'幅を{new_width - dims["width"]:.0f}mm広げる'
                    }
                ))

        elif category == 'balance':
            # バランス改善
            suggestions.append(ImprovementSuggestion(
                category='balance',
                priority=1,
                issue=issue,
                suggestion='脚の配置を見直す',
                expected_improvement=0.95,
                specific_action='重心が底面の中央に来るよう脚を再配置',
                estimated_values={
                    'method': '脚の位置を対称に配置 / 片側に脚を追加',
                    'check': '重心位置を底面の中央に収める'
                }
            ))

            # カウンターウェイト
            suggestions.append(ImprovementSuggestion(
                category='balance',
                priority=2,
                issue=issue,
                suggestion='カウンターウェイトを追加',
                expected_improvement=0.7,
                specific_action='反対側に重りを追加してバランスを取る',
                estimated_values={
                    'method': '重心と反対側の脚部に重りを追加',
                    'estimated_weight_kg': '2-5kg（サイズによる）'
                }
            ))

        elif category == 'general':
            # 一般的な提案
            suggestions.append(ImprovementSuggestion(
                category='general',
                priority=2,
                issue=issue,
                suggestion='複合的な改善を検討',
                expected_improvement=0.6,
                specific_action='複数の要素を組み合わせて改善',
                estimated_values={
                    'recommendations': [
                        '脚を太くする',
                        '接合部を強化する',
                        '材質をより剛性の高いものに変更',
                        '壁面固定を検討'
                    ]
                }
            ))

    # 優先度順にソート
    suggestions.sort(key=lambda x: (x.priority, -x.expected_improvement))

    return suggestions


# =============================================================================
# 改善効果の推定
# =============================================================================

def estimate_improved_stability(
    physics: Dict,
    suggestions: List[ImprovementSuggestion]
) -> Tuple[float, str]:
    """
    改善提案を実施した場合の安定性向上を推定

    Returns:
        (estimated_new_score, summary)
    """
    current_score = physics.get('stability_score', 0)

    # 各提案の改善効果を合算（重複考慮）
    improvements = {}
    for s in suggestions:
        if s.category not in improvements:
            improvements[s.category] = s.expected_improvement
        else:
            # 同カテゴリは最大値を採用
            improvements[s.category] = max(improvements[s.category], s.expected_improvement)

    total_improvement = sum(improvements.values()) * 0.5  # 控えめに見積もり
    estimated_score = min(current_score + total_improvement, 2.5)

    if estimated_score >= 2.0:
        summary = "提案を実施すれば安定性が大幅に向上する見込みです"
    elif estimated_score >= 1.5:
        summary = "提案を実施すれば安定性の改善が期待できます"
    else:
        summary = "根本的な設計変更が必要かもしれません"

    return estimated_score, summary


# =============================================================================
# メイン分析関数
# =============================================================================

def analyze_design(obj_path: Path, predictor: Optional['FurnitureStabilityPredictor'] = None) -> DesignAnalysis:
    """
    OBJファイルを分析し、設計改善提案を生成

    Args:
        obj_path: OBJファイルパス
        predictor: 予測器インスタンス（再利用する場合）
    """
    obj_path = Path(obj_path)

    if not obj_path.exists():
        raise FileNotFoundError(f"File not found: {obj_path}")

    # 安定性予測
    if predictor is None:
        predictor = FurnitureStabilityPredictor(verbose=False)
    result = predictor.predict(obj_path, use_tta=True)

    # 頂点データ取得
    vertices = load_obj_vertices(obj_path)

    # 物理特徴
    physics = result.get('physics', compute_physics_features(vertices))

    # 問題分析
    issues = analyze_issues(physics, result['prediction'])

    # 改善提案生成
    suggestions = generate_suggestions(physics, issues, vertices)

    # 改善効果推定
    estimated_score, summary = estimate_improved_stability(physics, suggestions)

    return DesignAnalysis(
        file_path=str(obj_path),
        furniture_type=result.get('furniture_type', 'unknown'),
        current_stability=result['prediction'],
        confidence=result['confidence'],
        physics=physics,
        issues=[issue for _, issue, _ in issues],
        suggestions=suggestions,
        estimated_improvement=estimated_score,
        summary=summary
    )


# =============================================================================
# 出力フォーマット
# =============================================================================

def print_analysis(analysis: DesignAnalysis):
    """分析結果を表示"""

    print("\n" + "=" * 70)
    print("  設計分析レポート")
    print("=" * 70)

    # 基本情報
    print(f"\nファイル: {Path(analysis.file_path).name}")
    print(f"家具タイプ: {analysis.furniture_type}")

    # 現在の状態
    if analysis.current_stability == 'stable':
        status = "\033[92m✓ 安定\033[0m"
    else:
        status = "\033[91m✗ 不安定\033[0m"

    print(f"\n【現在の状態】")
    print(f"  判定: {status}")
    print(f"  信頼度: {analysis.confidence * 100:.1f}%")
    print(f"  安定性スコア: {analysis.physics.get('stability_score', 0):.2f} / 2.5")

    # 物理特徴
    print(f"\n【物理特徴】")
    print(f"  重心高さ: {analysis.physics.get('relative_cog_height', 0) * 100:.0f}%")
    print(f"  アスペクト比: {analysis.physics.get('aspect_ratio', 0):.2f}")
    print(f"  重心位置: {'底面内' if analysis.physics.get('cog_over_base') else '底面外 ⚠'}")

    # 問題点
    if analysis.issues:
        print(f"\n【検出された問題】")
        for i, issue in enumerate(analysis.issues, 1):
            print(f"  {i}. {issue}")
    else:
        print(f"\n【問題なし】設計は良好です")

    # 改善提案
    if analysis.suggestions:
        print(f"\n" + "-" * 70)
        print("  改善提案")
        print("-" * 70)

        for i, s in enumerate(analysis.suggestions, 1):
            priority_mark = "★" * (4 - s.priority)
            print(f"\n  [{i}] {s.suggestion} {priority_mark}")
            print(f"      問題: {s.issue}")
            print(f"      対策: {s.specific_action}")
            print(f"      期待効果: {s.expected_improvement * 100:.0f}%向上")

            if s.estimated_values:
                print(f"      詳細:")
                for k, v in s.estimated_values.items():
                    if isinstance(v, list):
                        print(f"        {k}:")
                        for item in v:
                            print(f"          - {item}")
                    else:
                        print(f"        {k}: {v}")

    # 改善後の見込み
    print(f"\n" + "-" * 70)
    print(f"【改善見込み】")
    print(f"  推定スコア: {analysis.physics.get('stability_score', 0):.2f} → {analysis.estimated_improvement:.2f}")
    print(f"  {analysis.summary}")

    print("\n" + "=" * 70)


def export_to_json(analysis: DesignAnalysis, output_path: Path):
    """分析結果をJSONにエクスポート"""
    data = {
        'file_path': analysis.file_path,
        'furniture_type': analysis.furniture_type,
        'current_stability': analysis.current_stability,
        'confidence': analysis.confidence,
        'physics': analysis.physics,
        'issues': analysis.issues,
        'suggestions': [asdict(s) for s in analysis.suggestions],
        'estimated_improvement': analysis.estimated_improvement,
        'summary': analysis.summary
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='家具設計改善提案システム',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  python3 design_advisor.py chair.obj
  python3 design_advisor.py unstable_table.obj --output report.json
  python3 design_advisor.py furniture/*.obj --batch
        """
    )
    parser.add_argument('input', nargs='*', help='OBJファイル（複数可）')
    parser.add_argument('--output', '-o', help='JSONレポート出力先')
    parser.add_argument('--batch', '-b', action='store_true',
                        help='バッチモード（サマリーのみ表示）')
    parser.add_argument('--unstable-only', '-u', action='store_true',
                        help='不安定な家具のみ表示')
    args = parser.parse_args()

    if not args.input:
        parser.print_help()
        print("\n使用例:")
        print("  python3 design_advisor.py model.obj")
        return

    results = []

    for input_path in args.input:
        path = Path(input_path)
        if not path.is_absolute():
            path = Path.cwd() / path

        if not path.exists():
            print(f"⚠ ファイルが見つかりません: {path}")
            continue

        try:
            analysis = analyze_design(path)
            results.append(analysis)

            if args.unstable_only and analysis.current_stability == 'stable':
                continue

            if args.batch:
                status = "✗" if analysis.current_stability == 'unstable' else "✓"
                issues_count = len(analysis.issues)
                suggestions_count = len(analysis.suggestions)
                print(f"{status} {path.name}: {issues_count}問題, {suggestions_count}提案")
            else:
                print_analysis(analysis)

        except Exception as e:
            print(f"⚠ エラー ({path.name}): {e}")

    # JSON出力
    if args.output and results:
        output_path = Path(args.output)
        if len(results) == 1:
            export_to_json(results[0], output_path)
        else:
            # 複数ファイルの場合はリストとして出力
            data = [asdict(r) if hasattr(r, '__dataclass_fields__') else {
                'file_path': r.file_path,
                'furniture_type': r.furniture_type,
                'current_stability': r.current_stability,
                'confidence': r.confidence,
                'physics': r.physics,
                'issues': r.issues,
                'suggestions': [asdict(s) for s in r.suggestions],
                'estimated_improvement': r.estimated_improvement,
                'summary': r.summary
            } for r in results]
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\nレポートを保存: {output_path}")

    # バッチサマリー
    if args.batch and len(results) > 1:
        unstable_count = sum(1 for r in results if r.current_stability == 'unstable')
        print(f"\n合計: {len(results)}ファイル, {unstable_count}件が不安定")


if __name__ == "__main__":
    main()
