# データ収集方策が決定的な場合と確率的な場合の比較

## 実行コマンド

```bash
poetry run python -m recommender_experiments.app.logging_policy_experiment
```

## コード説明

まず、擬似的なバンディットデータセットを生成するためのコードを用意しましょう。

obpでは、`SyntheticBanditDataset`クラスを用いて、擬似的なバンディットデータセットを生成することができます。
その際に、`behavior_policy_function`引数に任意のデータ収集方策を設定することができます。

今回は決定的なデータ収集方策と確率的なデータ収集方策を比較したいので、まずはそれらのデータ収集方策の振る舞いを関数で定義します。

以下は、決定的(deterministic)なデータ収集方策を定義する関数です。

```python
def logging_policy_function_deterministic(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: int = None,
) -> np.ndarray:
    """(アクションa, 文脈x)の各組み合わせに対して、選択確率 π(a|x) を定義する関数。
    - 返り値のshape: (n_rounds, n_actions)
    - 今回は、決定論的に期待報酬の推定値 \hat{q}(x,a) が最大となるアクションを選択するデータ収集方策を設定
    - \hat{q}(x,a) は、今回はcontextによらず、事前に設定した固定の値とする
    - ニュース0: 0.05, ニュース1: 0.1, ニュース2: 0.15, ニュース3: 0.2
    - つまり任意の文脈xに対して、常にニュース4を選択するデータ収集方策を設定
    """
    n_rounds = context.shape[0]
    n_actions = action_context.shape[0]

    # 固定のスコアを設定 (n_actions=4として固定値を設定)
    fixed_scores = np.array([0.05, 0.1, 0.15, 0.2])

    # スコアが最大のアクションを確率1で選択する
    p_scores = np.array([0.0, 0.0, 0.0, 1.0])

    # 返り値の形式に整形: (n_rounds, n_actions)の配列で、各行が各ラウンドでのアクションの選択確率を表す
    action_dist = np.zeros((n_rounds, n_actions))
    action_dist[:, :] = p_scores

    assert np.allclose(
        action_dist.sum(axis=1), 1.0
    ), "各ラウンドでの全てのアクションの選択確率の合計は1.0である必要があります"

    return action_dist
```

続いて、確率的(stochastic)なデータ収集方策を定義する関数です。

```python
def logging_policy_function_stochastic(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: int = None,
) -> np.ndarray:
    """(アクションa, 文脈x)の各組み合わせに対して、選択確率 π(a|x) を定義する関数
    - 今回は、確率的なデータ収集方策を設定する。
    - 返り値のshape: (n_rounds, n_actions)
    """
    n_rounds = context.shape[0]
    n_actions = action_context.shape[0]

    # 文脈xによらない固定の選択確率を設定する
    p_scores = np.array([0.1, 0.2, 0.3, 0.4])

    # 返り値の形式に整形: (n_rounds, n_actions)の配列で、各行が各ラウンドでのアクションの選択確率を表す
    action_dist = np.zeros((n_rounds, n_actions))
    action_dist[:, :] = p_scores

    assert np.allclose(
        action_dist.sum(axis=1), 1.0
    ), "各ラウンドでの全てのアクションの選択確率の合計は1.0である必要があります"
    return action_dist
```

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
