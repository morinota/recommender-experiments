import numpy as np


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
    p_scores = np.array([0.01, 0.01, 0.01, 0.97])

    # 返り値の形式に整形: (n_rounds, n_actions)の配列で、各行が各ラウンドでのアクションの選択確率を表す
    action_dist = np.zeros((n_rounds, n_actions))
    action_dist[:, :] = p_scores

    assert np.allclose(
        action_dist.sum(axis=1), 1.0
    ), "各ラウンドでの全てのアクションの選択確率の合計は1.0である必要があります"
    return action_dist
