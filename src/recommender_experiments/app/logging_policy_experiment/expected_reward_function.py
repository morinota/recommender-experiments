import numpy as np


def expected_reward_function(context: np.ndarray, action_context: np.ndarray, random_state: int = None) -> np.ndarray:
    """(アクションa, 文脈x)の各組み合わせに対する期待報酬 E[r|x,a] を定義する関数
    今回の場合は、推薦候補4つの記事を送った場合の報酬rの期待値を、文脈xに依存しない固定値として設定する
    ニュース0: 0.2, ニュース1: 0.15, ニュース2: 0.1, ニュース3: 0.05
    返り値のshape: (n_rounds, n_actions, len_list)
    """
    n_rounds = context.shape[0]
    n_actions = action_context.shape[0]

    # 固定の期待報酬を設定 (n_actions=4として固定値を設定)
    fixed_rewards = np.array([0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.01, 0.001, 0.0001, 0.00001])

    # 文脈の数だけ期待報酬を繰り返して返す
    return np.tile(fixed_rewards, (n_rounds, 1))
