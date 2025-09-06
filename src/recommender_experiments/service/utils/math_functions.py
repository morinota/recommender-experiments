"""汎用的な数学関数とバンディット関連のユーティリティ関数."""

import numpy as np
from scipy.stats import rankdata
from sklearn.utils import check_random_state


def sigmoid(x: np.ndarray) -> np.ndarray:
    """シグモイド関数.

    Parameters
    ----------
    x : np.ndarray
        入力配列

    Returns
    -------
    np.ndarray
        シグモイド関数の出力
    """
    return np.exp(np.minimum(x, 0)) / (1.0 + np.exp(-np.abs(x)))


def softmax(x: np.ndarray) -> np.ndarray:
    """ソフトマックス関数.

    Parameters
    ----------
    x : np.ndarray
        入力配列 (n_samples, n_features)

    Returns
    -------
    np.ndarray
        ソフトマックス関数の出力
    """
    b = np.max(x, axis=1)[:, np.newaxis]
    numerator = np.exp(x - b)
    denominator = np.sum(numerator, axis=1)[:, np.newaxis]
    return numerator / denominator


def eps_greedy_policy(
    q_func: np.ndarray,
    eps: float = 0.5,
) -> np.ndarray:
    """epsilon-greedy法により方策を定義する.

    Parameters
    ----------
    q_func : np.ndarray
        Q関数の値 (n_samples, n_actions)
    eps : float, default=0.5
        ランダム行動を選ぶ確率

    Returns
    -------
    np.ndarray
        ε-greedy方策 (n_samples, n_actions)
    """
    is_topk = rankdata(-q_func, axis=1) <= 1
    pi = (1.0 - eps) * is_topk
    pi += eps / q_func.shape[1]
    return pi


def sample_action_fast(pi: np.ndarray, random_state: int = 12345) -> np.ndarray:
    """与えられた方策に従い、行動を高速に抽出する.

    Parameters
    ----------
    pi : np.ndarray
        行動選択確率 (n_samples, n_actions)
    random_state : int, default=12345
        乱数シード

    Returns
    -------
    np.ndarray
        サンプリングされた行動インデックス (n_samples,)
    """
    random_ = check_random_state(random_state)
    uniform_rvs = random_.uniform(size=pi.shape[0])[:, np.newaxis]
    cum_pi = pi.cumsum(axis=1)
    flg = cum_pi > uniform_rvs
    sampled_actions = flg.argmax(axis=1)
    return sampled_actions
