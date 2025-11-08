## Python開発方法

```
# テストの実行(単体・統合テスト全体)
uv run pytest
# 特定のテストファイルのみ実行
uv run pytest tests/test_ファイル名.py

# ruff formatでコードフォーマット (src/ と tests/ をそれぞれ指定すること)
uv run ruff format src/
uv run ruff format tests/
# importの自動整理 (src/ と tests/ をそれぞれ指定すること)
uv run ruff check --select I --fix src/
uv run ruff check --select I --fix tests/
```

## 単体テストのスタイルについて

- Arrange, Act, Assertの3段階に分けてテストコードを書くこと。
  - テストコード内で、`# Arrange`, `# Act`, `# Assert`のコメントを入れて、3段階を明示すること。
- テスト対象のクラスの変数名は `sut` (system under test) とすること。
- テストクラスを作らずに、テスト関数を作成すること。
- テスト関数名は、日本語で、動作を保証する「観察可能な振る舞い」のfactを表現すること。
  - 例: `test_指定された設定をもとにランキングタスクの合成ログデータが生成されること()`
- テストのassertionは、`assert <条件> "{保証するfactの内容}"`の形式で記述すること。

## 実装のスタイルについて

- 基本的にはPEP8に従うように記述すること。
- 引数名と変数名が一致する場合は、基本的には位置引数で渡すこと(キーワード引数にしても情報量が増えないため)。
  - 例: `self.sut.select_actions(context, available_actions, k)`

## marimoを使った開発方法

### 基本方針
- **探索・分析はmarimoノートブック**、**再利用可能なコードはsrc/モジュール**
- Jupyterではなくmarimoを使用（Gitフレンドリー、リアクティブ実行）
- ノートブックは純粋なPythonファイルとして管理

### marimoコマンド
```bash
# ノートブックの編集
uv run marimo edit notebooks/explore.py

# ファイル監視付き編集（外部エディタとの連携）
uv run marimo edit notebooks/explore.py --watch

# アプリとして実行（コード非表示）
uv run marimo run notebooks/report.py

# スクリプトとして実行（自動化用）
uv run python notebooks/analysis.py

# 引数付き実行
uv run marimo run notebooks/report.py -- --date 2024-01-01
uv run python notebooks/report.py --date 2024-01-01

# HTML形式でエクスポート
uv run marimo export html notebooks/report.py -o report.html
```

### marimoベストプラクティス
- **小さく集中したセル**で構成
- **リアクティブ実行**を活用（手動実行は不要）
- **UIエレメントには必ずlabelを設定**
- **`mo.stop()`で条件付き実行**を制御
- **重い処理はキャッシュ**を活用

### データ表示・可視化
```python
# インタラクティブデータフレーム
mo.ui.dataframe(df, page_size=10)

# Plotlyチャート（そのまま表示）
import plotly.express as px
chart = px.bar(df, ...)

# インタラクティブ選択にはAltairを推奨
mo.ui.altair_chart(...)
```

### スクリプト実行パターン
```python
import marimo as mo

if mo.running_in_notebook():
    # marimo edit/run時のデフォルト値
    date = "2024-01-01"
else:
    # スクリプト実行時はコマンドライン引数から取得
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    date = args.date
```
