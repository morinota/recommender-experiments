"""cfml_book全体で共通して使用されるユーティリティ関数"""

import numpy as np
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
