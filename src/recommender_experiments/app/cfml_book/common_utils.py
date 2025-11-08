"""cfml_book全体で共通して使用されるユーティリティ関数"""

import numpy as np
from scipy.stats import rankdata
from sklearn.utils import check_random_state


def sample_action_fast(pi: np.ndarray, random_state: int = 12345) -> np.ndarray:
    """与えられた方策に従い、行動を高速に抽出する."""
    random_ = check_random_state(random_state)
    uniform_rvs = random_.uniform(size=pi.shape[0])[:, np.newaxis]
    cum_pi = pi.cumsum(axis=1)
    flg = cum_pi > uniform_rvs
    sampled_actions = flg.argmax(axis=1)
    return sampled_actions


def sigmoid(x: np.ndarray) -> np.ndarray:
    """シグモイド関数."""
    return np.exp(np.minimum(x, 0)) / (1.0 + np.exp(-np.abs(x)))


def softmax(x: np.ndarray) -> np.ndarray:
    """ソフトマックス関数."""
    b = np.max(x, axis=1)[:, np.newaxis]
    numerator = np.exp(x - b)
    denominator = np.sum(numerator, axis=1)[:, np.newaxis]
    return numerator / denominator


def eps_greedy_policy(
    q_func: np.ndarray,
    k: int = 1,
    eps: float = 0.1,
    return_normalized: bool = True,
    rank_method: str | None = "ordinal",
    add_newaxis: bool = False,
) -> np.ndarray:
    """epsilon-greedy法により方策を定義する.

    期待報酬が高いtop-k個の行動に(1-eps)/kの確率を割り当て、
    すべての行動にeps/|A|の確率を加算する。

    Args:
        q_func: 期待報酬関数 (num_data, num_actions)
        k: 選択する上位k個の行動数
        eps: ランダム探索の確率（0から1の範囲）
        return_normalized: Trueの場合、確率の総和が1になるよう正規化する
        rank_method: rankdataのmethod引数. "ordinal", "average", "dense"など.
                     Noneの場合はデフォルト動作(average)
        add_newaxis: Trueの場合、最後の次元に新しい軸を追加 (num_data, num_actions, 1)

    Returns:
        epsilon-greedy方策の行動選択確率分布
        - add_newaxis=False: (num_data, num_actions)
        - add_newaxis=True: (num_data, num_actions, 1)

    Examples:
        >>> q_func = np.array([[1.0, 0.5, 0.3], [0.8, 0.9, 0.2]])
        >>> # Top-1選択、eps=0.1
        >>> pi = eps_greedy_policy(q_func, k=1, eps=0.1)
        >>> pi.shape
        (2, 3)
        >>> np.allclose(pi.sum(axis=1), 1.0)
        True
        >>> # 新しい軸を追加
        >>> pi_with_axis = eps_greedy_policy(q_func, k=1, eps=0.1, add_newaxis=True)
        >>> pi_with_axis.shape
        (2, 3, 1)
    """
    if rank_method is not None:
        is_topk = rankdata(-q_func, method=rank_method, axis=1) <= k
    else:
        is_topk = rankdata(-q_func, axis=1) <= k

    pi = ((1.0 - eps) / k) * is_topk + eps / q_func.shape[1]

    if return_normalized:
        pi = pi / pi.sum(1)[:, np.newaxis]

    if add_newaxis:
        pi = pi[:, :, np.newaxis]

    return pi
