# Furniture Stability & Load Capacity AI

3D家具モデルの安定性と耐荷重を予測するAIシステム

## Overview

点群データから家具の物理的特性を判定するAIモデルです。

### 安定性予測
- **精度**: 97.9%
- **対応家具**: 8種類 (chair, table, shelf, cabinet, desk, sofa, stool, bench)
- **学習データ**: 11,396サンプル

### 耐荷重予測
- **精度**: 98.5%（材質指定時）
- **対応**: 棚板の静荷重・動荷重
- **材質**: 8種類対応

## Features

- PointNet + Local Features アーキテクチャ
- Test-Time Augmentation (TTA) による高精度推論
- 物理特徴（重心位置、底面積、アスペクト比）の解析
- 梁理論に基づく耐荷重計算
- 動荷重対応（衝撃係数2.0）
- 3D可視化ツール
- 対話式材質選択UI
- **設計改善提案システム** - 不安定な家具に対する具体的な改善案
- **バッチ解析・レポート生成** - HTML/CSV/JSON形式でレポート出力
- **Blenderアドオン** - Blender内で直接解析実行

## Installation

```bash
# Clone
git clone https://github.com/tsuyoshi-sk/furniture-stability-ai.git
cd furniture-stability-ai

# Dependencies
pip install torch numpy matplotlib
```

## Usage

### Unified Analysis (統合解析)

```bash
# Interactive mode (材質選択UI付き)
python3 scripts/utils/furniture_analyzer.py shelf.obj

# With material specified
python3 scripts/utils/furniture_analyzer.py shelf.obj --material plywood

# List available materials
python3 scripts/utils/furniture_analyzer.py --list-materials
```

**Output example:**
```
============================================================
解析結果
============================================================

ファイル: shelf_board.obj
材質: 合板

------------------------------------------------------------
【安定性】
------------------------------------------------------------
  判定: ✓ 安定
  信頼度: 97.0%
  家具タイプ: shelf

------------------------------------------------------------
【耐荷重】
------------------------------------------------------------
  静荷重上限: 30.7 kg
  動荷重上限: 15.3 kg
  安全率: 2.5
  信頼度: 100%

------------------------------------------------------------
【推奨事項】
------------------------------------------------------------
  △ 一般的な書籍程度まで対応可能です。
```

### Stability Inference (安定性予測)

```bash
# Single file
python3 scripts/inference/inference.py model.obj

# Multiple files
python3 scripts/inference/inference.py file1.obj file2.obj file3.obj

# Evaluate all types
python3 scripts/inference/inference.py --evaluate
```

### Load Capacity (耐荷重予測)

```bash
# With material
python3 scripts/utils/load_capacity.py shelf.obj --material oak

# List materials
python3 scripts/utils/load_capacity.py --list-materials

# With custom safety factor
python3 scripts/utils/load_capacity.py shelf.obj -m plywood --safety-factor 3.0
```

### Visualization (可視化)

```bash
# Single file with details
python3 scripts/utils/visualize.py model.obj

# Compare multiple files
python3 scripts/utils/visualize.py stable.obj unstable.obj --compare

# Generate gallery for all furniture types
python3 scripts/utils/visualize.py --gallery
```

### Design Advisor (設計改善提案)

不安定な家具に対して具体的な改善提案を生成します。

```bash
# 単一ファイル解析
python3 scripts/utils/design_advisor.py chair.obj

# 複数ファイルをバッチ処理
python3 scripts/utils/design_advisor.py *.obj --batch

# 不安定なもののみ表示
python3 scripts/utils/design_advisor.py *.obj --batch --unstable-only

# JSON出力
python3 scripts/utils/design_advisor.py chair.obj --output report.json
```

**Output example:**
```
======================================================================
  設計分析レポート
======================================================================

ファイル: chair.obj
家具タイプ: chair

【現在の状態】
  判定: ✗ 不安定
  信頼度: 95.2%
  安定性スコア: 1.10 / 2.5

【検出された問題】
  1. 底面積が小さい
  2. 重心位置がやや高い

----------------------------------------------------------------------
  改善提案
----------------------------------------------------------------------

  [1] 脚の間隔を広げる ★★★
      問題: 底面積が小さい
      対策: 脚間隔を50mm広げる
      期待効果: 54%向上

  [2] 重心位置を下げる ★★
      問題: 重心位置がやや高い
      対策: 上部を軽量化するか、下部に重量を追加
      期待効果: 11%向上
```

### Batch Analyzer (バッチ解析・レポート生成)

複数ファイルを一括解析し、HTML/CSV/JSONレポートを生成します。

```bash
# 基本的な使用法
python3 scripts/utils/batch_analyzer.py data/*.obj --output report

# 上限を指定
python3 scripts/utils/batch_analyzer.py *.obj --limit 100

# 出力形式を指定
python3 scripts/utils/batch_analyzer.py *.obj --format html,csv

# 静かモード
python3 scripts/utils/batch_analyzer.py *.obj --quiet
```

生成されるレポート:
- `report.html` - ブラウザで閲覧可能なインタラクティブレポート
- `report.csv` - Excel等で開けるCSVファイル
- `report.json` - プログラムから利用可能なJSONデータ

### Blender Addon

Blender内で直接家具の安定性解析を実行できます。

**インストール:**
1. Blender → Edit → Preferences → Add-ons
2. "Install..." → `scripts/blender/furniture_stability_addon.py` を選択
3. "Furniture Stability Analyzer" を有効化

**使用方法:**
1. 3D Viewport右側パネル（Nキー）→ "Furniture" タブ
2. Python PathとScripts Pathを設定
3. メッシュを選択 → "Analyze Stability"

詳細は `scripts/blender/README.md` を参照してください。

## Supported Materials

| # | Material | Japanese | E (MPa) | Strength | Typical Use |
|---|----------|----------|---------|----------|-------------|
| 1 | pine | パイン材 | 10,000 | Medium | DIY furniture |
| 2 | oak | オーク材 | 12,000 | High | Premium furniture |
| 3 | plywood | 合板 | 8,000 | Medium | General shelves |
| 4 | mdf | MDF | 3,500 | Low-Med | Color boxes |
| 5 | particle_board | パーティクルボード | 2,500 | Low | Budget furniture |
| 6 | steel | スチール | 200,000 | Very High | Metal racks |
| 7 | aluminum | アルミニウム | 70,000 | High | Outdoor furniture |
| 8 | plastic_abs | ABS樹脂 | 2,300 | Low-Med | Plastic furniture |

## Project Structure

```
furniture-stability-ai/
├── scripts/
│   ├── training/                    # Training scripts
│   │   ├── train_augmented.py       # Data augmentation training
│   │   ├── train_unified.py         # Unified model training
│   │   ├── train_ensemble*.py       # Ensemble models
│   │   ├── train_load_capacity*.py  # Load capacity models
│   │   └── train_point*.py          # PointNet2/Transformer
│   ├── inference/                   # Inference scripts
│   │   ├── inference.py             # Basic inference
│   │   ├── inference_ensemble.py    # Ensemble inference
│   │   └── predict_*.py             # Prediction utilities
│   ├── utils/                       # Utility scripts
│   │   ├── furniture_analyzer.py    # Unified analysis tool
│   │   ├── design_advisor.py        # Design improvement advisor
│   │   ├── batch_analyzer.py        # Batch analysis & reporting
│   │   ├── load_capacity.py         # Load capacity prediction
│   │   ├── visualize.py             # 3D visualization
│   │   ├── generate_furniture.py    # Dataset generation
│   │   └── bracket_database.py      # Bracket/hardware database
│   └── blender/                     # Blender integration
│       ├── furniture_stability_addon.py  # Blender addon
│       └── README.md                # Addon documentation
├── models/                          # Trained models
│   ├── models_augmented/            # Main stability model
│   ├── models_unified/              # Unified model
│   └── models_load/                 # Load capacity model
├── data/
│   └── datasets/                    # Training datasets
│       ├── dataset/                 # Stability data (stable/unstable)
│       ├── dataset_unified/         # Unified training data
│       └── dataset_load_v2/         # Load capacity data
├── output/                          # Generated outputs
│   ├── visualizations/              # Generated images
│   ├── drawings/                    # Technical drawings
│   └── images/                      # Rendered images
└── docs/                            # Documentation & catalogs
```

## Model Performance

### Stability Prediction

| Furniture Type | Stable | Unstable | Total |
|----------------|--------|----------|-------|
| chair          | 86.0%  | 96.0%    | 91.0% |
| table          | 92.0%  | 92.0%    | 92.0% |
| shelf          | 100%   | 100%     | 100%  |
| cabinet        | 100%   | 100%     | 100%  |
| desk           | 100%   | 100%     | 100%  |
| sofa           | 100%   | 100%     | 100%  |
| stool          | 100%   | 100%     | 100%  |
| bench          | 100%   | 100%     | 100%  |
| **Overall**    |        |          | **97.9%** |

### Load Capacity Prediction

| Condition | Mean Error | Within 5% |
|-----------|------------|-----------|
| With correct material | 0.25% | 98.5% |
| Auto-estimated material | 44% | 34.5% |

## Visualization Examples

### Single File View
![Sofa Visualization](output/visualizations/demo_sofa.png)

### Gallery View
![Desk Gallery](output/visualizations/gallery_desk.png)

## Technical Details

### Stability Model Architecture
- **Encoder**: PointNet with Local Features (512-dim)
- **Classifier**: 3-layer MLP with BatchNorm and Dropout
- **Input**: 1024 sampled points from OBJ mesh

### Training
- Data Augmentation: Rotation, Scale, Noise, Dropout, Jitter, Shift, Mirror
- Mixup: alpha=0.2
- Label Smoothing: 0.1
- Optimizer: AdamW with Cosine Annealing
- Early Stopping: patience=30

### Load Capacity Calculation
- **Theory**: Euler-Bernoulli beam theory
- **Limits**: Stress limit AND deflection limit (L/200)
- **Dynamic Factor**: 2.0x for impact loads
- **Safety Factor**: 2.5 (default)

### Physics Features
- **CoG Height**: Center of gravity relative height (0=bottom, 1=top)
- **Base Area**: Support polygon area
- **Aspect Ratio**: Height / Base width
- **CoG over Base**: Whether CoG projects within base
- **Moment of Inertia**: I = bh³/12
- **Section Modulus**: S = bh²/6

## API Usage (Python)

```python
from scripts.utils.furniture_analyzer import analyze_furniture

# Analyze with material
results = analyze_furniture(
    "shelf.obj",
    material="plywood",
    interactive=False
)

print(f"Stability: {results['stability']['prediction']}")
print(f"Max Load: {results['load_capacity']['max_dynamic_load_kg']:.1f} kg")
```

## License

MIT License

## Author

Created with Claude Code
