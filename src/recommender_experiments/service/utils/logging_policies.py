import numpy as np


def random_policy(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: int = None,
) -> np.ndarray:
    """一様ランダムにアイテムを選択するデータ収集方策"""
    n_rounds = context.shape[0]
    n_actions = action_context.shape[0]

    action_dist = np.full((n_rounds, n_actions), 1.0 / n_actions)

    assert action_dist.shape == (n_rounds, n_actions)
    return action_dist


def context_free_determinisitic_policy(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: int = None,
) -> np.ndarray:
    """コンテキストを考慮せず、全てのcontextに対して必ず最後尾のアイテム $a_{|A|-1}$ を推薦する決定的方策"""
    n_rounds = context.shape[0]
    n_actions = action_context.shape[0]

    # contextによらず、最後尾のアクションを確率1.0で決定的に選択
    action_dist = np.full((n_rounds, n_actions), 0.0)
    last_action_idx = n_actions - 1
    action_dist[:, last_action_idx] = 1.0

    assert action_dist.shape == (n_rounds, n_actions)
    return action_dist


def context_free_stochastic_policy(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: int = None,
) -> np.ndarray:
    """コンテキストを考慮なしの確率的方策(epsilon-greedy)
    - epsilon=0.1とする。
    - 活用フェーズでは、必ず最後尾のアイテム $a_{|A|-1}$ を選択する。
    """
    n_rounds = context.shape[0]
    n_actions = action_context.shape[0]
    epsilon = 0.1

    # contextによらず、確率epsilonで全てのアクションを一様ランダムに選択
    action_dist = np.full((n_rounds, n_actions), epsilon / n_actions)
    # 確率1-epsilonで最後尾のアクションを決定的に選択
    last_action_idx = n_actions - 1
    action_dist[np.arange(n_rounds), last_action_idx] += 1.0 - epsilon

    assert action_dist.shape == (n_rounds, n_actions)
    return action_dist


def context_aware_determinisitic_policy(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: int = None,
) -> np.ndarray:
    """ユーザとアイテムのコンテキストを考慮し、
    コンテキストベクトル $x$ とアイテムコンテキストベクトル $e$ の内積が最も大きいアイテムを
    確率1.0で推薦する決定的方策。
    返り値:
        action_dist: 推薦確率 (shape: (n_rounds, n_actions))
    """
    n_rounds = context.shape[0]
    n_actions = action_context.shape[0]

    # 内積を計算
    scores = context @ action_context.T  # shape: (n_rounds, n_actions)

    # 各ラウンドで最もスコアが高いアクションのindexを取得
    selected_actions = np.argmax(scores, axis=1)  # shape: (n_rounds,)

    # 確率1.0で最もスコアが高いアクションを決定的に選択
    action_dist = np.full((n_rounds, n_actions), 0.0)
    action_dist[np.arange(n_rounds), selected_actions] = 1.0

    assert action_dist.shape == (n_rounds, n_actions)
    return action_dist


def context_aware_stochastic_policy(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: int = None,
) -> np.ndarray:
    """ユーザとアイテムのコンテキストを考慮し、
    コンテキストベクトル $x$ とアイテムコンテキストベクトル $e$ の内積が最も大きいアイテムを
    確率1-epsilonで推薦し、その他のアイテムを一様ランダムに確率epsilonで推薦する確率的方策。
    返り値:
        action_dist: 推薦確率 (shape: (n_rounds, n_actions))
    """
    n_rounds = context.shape[0]
    n_actions = action_context.shape[0]
    epsilon = 0.1

    # 内積を計算
    scores = context @ action_context.T  # shape: (n_rounds, n_actions)

    # 各ラウンドで最もスコアが高いアクションのindexを取得
    selected_actions = np.argmax(scores, axis=1)  # shape: (n_rounds,)

    # 確率的方策: 確率epsilonで全てのアクションを一様ランダムに選択し、確率1-epsilonで最もスコアが高いアクションを決定的に選択
    action_dist = np.full((n_rounds, n_actions), epsilon / n_actions)
    action_dist[np.arange(n_rounds), selected_actions] += 1.0 - epsilon

    assert action_dist.shape == (n_rounds, n_actions)
    return action_dist
