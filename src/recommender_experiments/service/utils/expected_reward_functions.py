# 実験用に、真の期待報酬関数 E_{p(r|x,a)}[r] を定義するモジュール


import numpy as np


def context_free_binary(
    context: np.ndarray,  # shape: (n_rounds, dim_context)
    action_context: np.ndarray,  # shape: (n_actions, dim_action_context)
    random_state: int = None,
    lower: float = 0.0,
    upper: float = 1.0,
) -> np.ndarray:  # (n_rounds, n_actions, len_list)
    """(アクションa, 文脈x)の各組み合わせに対する期待報酬 E_{p(r|x,a)}[r] を定義する関数。
    今回は、文脈xに依存しない、アクション毎に固定のcontext-freeな期待報酬関数を想定している。
    具体的には、アクションaのindexが0から大きくなるにつれて、期待報酬がupperからlowerに線形に減少するような関数を想定している。
    Args:
        context (np.ndarray): 文脈x。 (n_rounds, dim_context)の配列。
        action_context (np.ndarray): アクション特徴量。 (n_actions, dim_action_context)の配列。
        random_state (int, optional): 乱数シード. Defaults to None.
        lower (float, optional): 期待値の下限値. Defaults to 0.0.
        upper (float, optional): 期待値の上限値. Defaults to 1.0.
    Returns:
        np.ndarray: 期待報酬 (n_rounds, n_actions, len_list) の配列。
    """
    # 乱数シードを設定
    if random_state is not None:
        np.random.seed(random_state)

    n_rounds = context.shape[0]
    n_actions = action_context.shape[0]
    len_list = 1  # len_listは1で固定とする

    # アクションaのindexが0から大きくなるにつれて、期待報酬がupperからlowerに線形に減少する配列を生成
    action_rewards = np.linspace(upper, lower, n_actions)
    print(action_rewards)

    # 各ラウンドに対して同じ期待報酬を繰り返す
    rewards = np.tile(action_rewards, (n_rounds, 1))

    return rewards


# 試しに期待報酬関数を実行してみる
n_rounds = 3
n_actions = 50
dim_context = 300
dim_action_context = 300
context = np.random.random((n_rounds, dim_context))
action_context = np.random.random((n_actions, dim_action_context))
print(context_free_binary(context, action_context))
