#!/usr/bin/env python3
"""
Batch Analyzer - 家具一括解析・レポート生成システム

複数のOBJファイルを一括解析し、HTML/CSV/JSONレポートを生成
"""
import sys
import json
import csv
import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import glob

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "inference"))

# 静かにインポート
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


@dataclass
class AnalysisResult:
    """解析結果"""
    file_name: str
    file_path: str
    furniture_type: str
    stability: str
    confidence: float
    stability_score: float
    cog_height: float
    aspect_ratio: float
    cog_in_base: bool
    issues_count: int
    suggestions_count: int
    top_issue: str
    top_suggestion: str


def analyze_files(file_paths: List[Path], quiet: bool = False) -> List[AnalysisResult]:
    """複数ファイルを解析"""
    from design_advisor import analyze_design
    from inference import FurnitureStabilityPredictor

    results = []
    total = len(file_paths)

    # 予測器を一度だけ初期化
    predictor = FurnitureStabilityPredictor(verbose=not quiet)

    for i, path in enumerate(file_paths, 1):
        if not quiet:
            print(f"\r解析中... {i}/{total} ({path.name})          ", end="", flush=True)

        try:
            analysis = analyze_design(path, predictor=predictor)

            top_issue = analysis.issues[0] if analysis.issues else ""
            top_suggestion = analysis.suggestions[0].suggestion if analysis.suggestions else ""

            result = AnalysisResult(
                file_name=path.name,
                file_path=str(path),
                furniture_type=analysis.furniture_type,
                stability=analysis.current_stability,
                confidence=analysis.confidence,
                stability_score=analysis.physics.get('stability_score', 0),
                cog_height=analysis.physics.get('relative_cog_height', 0),
                aspect_ratio=analysis.physics.get('aspect_ratio', 0),
                cog_in_base=analysis.physics.get('cog_over_base', True),
                issues_count=len(analysis.issues),
                suggestions_count=len(analysis.suggestions),
                top_issue=top_issue,
                top_suggestion=top_suggestion
            )
            results.append(result)

        except Exception as e:
            if not quiet:
                print(f"\n⚠ エラー ({path.name}): {e}")

    if not quiet:
        print()

    return results


def generate_summary(results: List[AnalysisResult]) -> Dict:
    """統計サマリーを生成"""
    if not results:
        return {}

    total = len(results)
    stable = sum(1 for r in results if r.stability == 'stable')
    unstable = total - stable

    # 家具タイプ別
    by_type = {}
    for r in results:
        ft = r.furniture_type
        if ft not in by_type:
            by_type[ft] = {'total': 0, 'stable': 0, 'unstable': 0}
        by_type[ft]['total'] += 1
        if r.stability == 'stable':
            by_type[ft]['stable'] += 1
        else:
            by_type[ft]['unstable'] += 1

    # よくある問題
    issue_counts = {}
    for r in results:
        if r.top_issue:
            issue_counts[r.top_issue] = issue_counts.get(r.top_issue, 0) + 1

    top_issues = sorted(issue_counts.items(), key=lambda x: -x[1])[:5]

    return {
        'total': total,
        'stable': stable,
        'unstable': unstable,
        'stable_rate': stable / total * 100 if total > 0 else 0,
        'by_type': by_type,
        'top_issues': top_issues,
        'avg_confidence': sum(r.confidence for r in results) / total if total > 0 else 0,
        'avg_stability_score': sum(r.stability_score for r in results) / total if total > 0 else 0,
    }


def export_csv(results: List[AnalysisResult], output_path: Path):
    """CSVにエクスポート"""
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            'ファイル名', 'パス', '家具タイプ', '安定性', '信頼度',
            '安定性スコア', '重心高さ', 'アスペクト比', '重心位置',
            '問題数', '提案数', '主な問題', '主な提案'
        ])
        for r in results:
            writer.writerow([
                r.file_name, r.file_path, r.furniture_type,
                '安定' if r.stability == 'stable' else '不安定',
                f'{r.confidence * 100:.1f}%',
                f'{r.stability_score:.2f}',
                f'{r.cog_height * 100:.0f}%',
                f'{r.aspect_ratio:.2f}',
                '底面内' if r.cog_in_base else '底面外',
                r.issues_count, r.suggestions_count,
                r.top_issue, r.top_suggestion
            ])


def export_json(results: List[AnalysisResult], summary: Dict, output_path: Path):
    """JSONにエクスポート"""
    data = {
        'generated_at': datetime.now().isoformat(),
        'summary': summary,
        'results': [asdict(r) for r in results]
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def export_html(results: List[AnalysisResult], summary: Dict, output_path: Path):
    """HTMLレポートを生成"""

    # 不安定なものを先に
    sorted_results = sorted(results, key=lambda x: (x.stability == 'stable', -x.issues_count))

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>家具安定性解析レポート</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #333; margin-bottom: 20px; }}
        .summary {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .summary h2 {{ color: #555; margin-bottom: 15px; font-size: 1.2em; }}
        .stats {{ display: flex; gap: 20px; flex-wrap: wrap; }}
        .stat {{ background: #f8f9fa; padding: 15px 20px; border-radius: 6px; min-width: 150px; }}
        .stat-value {{ font-size: 2em; font-weight: bold; color: #333; }}
        .stat-label {{ color: #666; font-size: 0.9em; }}
        .stat.stable .stat-value {{ color: #28a745; }}
        .stat.unstable .stat-value {{ color: #dc3545; }}
        .issues {{ margin-top: 15px; }}
        .issue-item {{ display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #eee; }}
        table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #333; color: white; font-weight: 500; }}
        tr:hover {{ background: #f8f9fa; }}
        .stable {{ color: #28a745; font-weight: bold; }}
        .unstable {{ color: #dc3545; font-weight: bold; }}
        .score {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.85em; }}
        .score-high {{ background: #d4edda; color: #155724; }}
        .score-mid {{ background: #fff3cd; color: #856404; }}
        .score-low {{ background: #f8d7da; color: #721c24; }}
        .footer {{ text-align: center; color: #666; margin-top: 20px; font-size: 0.9em; }}
        .type-stats {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px; }}
        .type-badge {{ background: #e9ecef; padding: 5px 10px; border-radius: 4px; font-size: 0.85em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>家具安定性解析レポート</h1>

        <div class="summary">
            <h2>サマリー</h2>
            <div class="stats">
                <div class="stat">
                    <div class="stat-value">{summary['total']}</div>
                    <div class="stat-label">総ファイル数</div>
                </div>
                <div class="stat stable">
                    <div class="stat-value">{summary['stable']}</div>
                    <div class="stat-label">安定</div>
                </div>
                <div class="stat unstable">
                    <div class="stat-value">{summary['unstable']}</div>
                    <div class="stat-label">不安定</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{summary['stable_rate']:.1f}%</div>
                    <div class="stat-label">安定率</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{summary['avg_stability_score']:.2f}</div>
                    <div class="stat-label">平均スコア</div>
                </div>
            </div>

            <div class="type-stats">
                <strong>家具タイプ別:</strong>
"""

    for ft, counts in summary['by_type'].items():
        rate = counts['stable'] / counts['total'] * 100 if counts['total'] > 0 else 0
        html += f'<span class="type-badge">{ft}: {counts["total"]}件 ({rate:.0f}%安定)</span>\n'

    html += """
            </div>

            <div class="issues">
                <strong>よくある問題:</strong>
"""

    for issue, count in summary['top_issues']:
        html += f'<div class="issue-item"><span>{issue}</span><span>{count}件</span></div>\n'

    html += """
            </div>
        </div>

        <table>
            <thead>
                <tr>
                    <th>ファイル名</th>
                    <th>タイプ</th>
                    <th>判定</th>
                    <th>信頼度</th>
                    <th>スコア</th>
                    <th>問題数</th>
                    <th>主な問題</th>
                </tr>
            </thead>
            <tbody>
"""

    for r in sorted_results:
        status_class = 'stable' if r.stability == 'stable' else 'unstable'
        status_text = '安定' if r.stability == 'stable' else '不安定'

        if r.stability_score >= 2.0:
            score_class = 'score-high'
        elif r.stability_score >= 1.0:
            score_class = 'score-mid'
        else:
            score_class = 'score-low'

        html += f"""
                <tr>
                    <td>{r.file_name}</td>
                    <td>{r.furniture_type}</td>
                    <td class="{status_class}">{status_text}</td>
                    <td>{r.confidence * 100:.1f}%</td>
                    <td><span class="score {score_class}">{r.stability_score:.2f}</span></td>
                    <td>{r.issues_count}</td>
                    <td>{r.top_issue or '-'}</td>
                </tr>
"""

    html += f"""
            </tbody>
        </table>

        <div class="footer">
            生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |
            Furniture Stability AI
        </div>
    </div>
</body>
</html>
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)


def print_summary(summary: Dict):
    """サマリーを表示"""
    print("\n" + "=" * 60)
    print("  解析サマリー")
    print("=" * 60)

    print(f"\n  総ファイル数: {summary['total']}")
    print(f"  安定: \033[92m{summary['stable']}\033[0m ({summary['stable_rate']:.1f}%)")
    print(f"  不安定: \033[91m{summary['unstable']}\033[0m")
    print(f"  平均スコア: {summary['avg_stability_score']:.2f} / 2.5")

    print(f"\n  【家具タイプ別】")
    for ft, counts in summary['by_type'].items():
        rate = counts['stable'] / counts['total'] * 100 if counts['total'] > 0 else 0
        bar = "█" * int(rate / 10) + "░" * (10 - int(rate / 10))
        print(f"    {ft:12} {bar} {rate:5.1f}% ({counts['stable']}/{counts['total']})")

    if summary['top_issues']:
        print(f"\n  【よくある問題】")
        for issue, count in summary['top_issues']:
            print(f"    - {issue}: {count}件")

    print("=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='家具一括解析・レポート生成',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  python3 batch_analyzer.py data/datasets/dataset/unstable/*.obj
  python3 batch_analyzer.py *.obj --output report
  python3 batch_analyzer.py furniture/ --format html,csv
        """
    )
    parser.add_argument('input', nargs='*', help='OBJファイルまたはディレクトリ')
    parser.add_argument('--output', '-o', default='report',
                        help='出力ファイル名（拡張子なし）')
    parser.add_argument('--format', '-f', default='html,csv,json',
                        help='出力形式 (html,csv,json)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='進捗を表示しない')
    parser.add_argument('--limit', '-l', type=int,
                        help='解析ファイル数の上限')
    args = parser.parse_args()

    if not args.input:
        parser.print_help()
        return

    # ファイルリストを構築
    file_paths = []
    for pattern in args.input:
        path = Path(pattern)
        if path.is_dir():
            file_paths.extend(path.glob('**/*.obj'))
        elif '*' in pattern:
            file_paths.extend(Path(p) for p in glob.glob(pattern))
        elif path.exists():
            file_paths.append(path)

    if not file_paths:
        print("⚠ 対象ファイルが見つかりません")
        return

    # 上限適用
    if args.limit:
        file_paths = file_paths[:args.limit]

    print(f"\n{len(file_paths)}ファイルを解析します...")

    # 解析実行
    results = analyze_files(file_paths, quiet=args.quiet)

    if not results:
        print("⚠ 解析結果がありません")
        return

    # サマリー生成
    summary = generate_summary(results)
    print_summary(summary)

    # 出力
    formats = [f.strip().lower() for f in args.format.split(',')]
    output_base = Path(args.output)

    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(exist_ok=True)

    if 'csv' in formats:
        csv_path = output_dir / f"{output_base.name}.csv"
        export_csv(results, csv_path)
        print(f"\n✓ CSV保存: {csv_path}")

    if 'json' in formats:
        json_path = output_dir / f"{output_base.name}.json"
        export_json(results, summary, json_path)
        print(f"✓ JSON保存: {json_path}")

    if 'html' in formats:
        html_path = output_dir / f"{output_base.name}.html"
        export_html(results, summary, html_path)
        print(f"✓ HTML保存: {html_path}")


if __name__ == "__main__":
    main()
