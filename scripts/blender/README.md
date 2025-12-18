# Furniture Stability Blender Addon

Blender内で家具の安定性解析と設計改善提案を行うアドオン

## インストール

### 1. アドオンをインストール

1. Blenderを開く
2. Edit > Preferences > Add-ons
3. "Install..." をクリック
4. `furniture_stability_addon.py` を選択
5. "Furniture Stability Analyzer" にチェックを入れて有効化

### 2. 設定

1. 3D Viewport の右側パネル（Nキー）を開く
2. "Furniture" タブを選択
3. 設定:
   - **Python Path**: PyTorchがインストールされたPython（例: `/usr/bin/python3`）
   - **Scripts Path**: `furniture_research/scripts` ディレクトリのパス

## 使い方

### 単一オブジェクト解析

1. メッシュオブジェクトを選択
2. "Analyze Stability" をクリック
3. 結果がパネルと通知に表示される

### 全オブジェクト解析

1. "Analyze All Objects" をクリック
2. シーン内のすべてのメッシュが解析される

### 不安定オブジェクトの選択

- "Highlight Unstable" で不安定なオブジェクトを選択

### 改善提案の表示

- "Show Suggestions" でポップアップ表示

## 結果の確認

解析結果はオブジェクトのカスタムプロパティに保存されます：

- `stability`: "stable" または "unstable"
- `confidence`: 信頼度 (0-1)
- `stability_score`: 安定性スコア (0-2.5)
- `issues`: 検出された問題のリスト
- `suggestions`: 改善提案のリスト

## 必要要件

- Blender 3.0以上
- Python 3.8以上（PyTorch, NumPy インストール済み）
- furniture_research プロジェクト

## トラブルシューティング

### "Scripts path not set" エラー

→ パネルで Scripts Path を設定してください

### "Analysis failed" エラー

→ Python Path が正しいか確認してください
→ ターミナルで `python3 scripts/utils/design_advisor.py --help` が動作するか確認

### 解析が遅い

→ 初回はモデル読み込みに時間がかかります（約10秒）
