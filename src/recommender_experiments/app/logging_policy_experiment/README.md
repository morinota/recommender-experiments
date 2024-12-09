# データ収集方策が決定的な場合と確率的な場合の比較

## 実行コマンド

```bash
poetry run python -m recommender_experiments.app.logging_policy_experiment
```

## コード説明

まず、擬似データを準備するコードを用意します。

```python
# ニュース推薦用の擬似的なバンディットデータセット (i.e. 文脈xにおいてアクションaを選択して、得られた報酬r)を設定する
dataset = SyntheticBanditDataset(
    n_actions=4,  # 推薦候補の数
    dim_context=5,  # 文脈ベクトルの次元数
    reward_type="binary",  # 報酬となるmetricsの種類("binary" or "continuous")
    reward_function=logistic_reward_function,  # 報酬の期待値 E_{(x,a)}[r|x,a] の真の値を設定
    beta=3.0,  # 決定的か確率的かを制御（後述）
    random_state=123,  # 再現性のためのランダムシード
)

# バンディットデータを生成
bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=10000)
```

ここで

- `n_actions`: 推薦候補の数
- `dim_context`: 文脈ベクトルの次元数
- `reward_type`: 報酬となるmetricsの種類(今回は、届いたプッシュ通知をもとにユーザが購読してくれるかどうかを表すバイナリ値)

- `reward_function`: 各（アクション、文脈）の組み合わせに対する期待報酬の真の値を返す関数。
