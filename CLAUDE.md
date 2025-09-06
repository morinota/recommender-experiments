## Python開発方法

```
# テストの実行(単体・統合テスト全体)
poetry run pytest
# 特定のテストファイルのみ実行
poetry run pytest tests/test_ファイル名.py

# ruff formatでコードフォーマット
poetry run ruff format .
```

## 単体テストのスタイルについて

- Arrange, Act, Assertの3段階に分けてテストコードを書くこと。
- テストクラスを作らずに、テスト関数を作成すること。
- テスト関数名は、日本語で、動作を保証する「観察可能な振る舞い」のfactを表現すること。
  - 例: `test_指定された設定をもとにランキングタスクの合成ログデータが生成されること()`
- テストのassertionは、`assert <条件> "{保証するfactの内容}"`の形式で記述すること。
