import collections
import itertools
from dataclasses import dataclass
from typing import Callable, TypedDict

import numpy as np
from obp.dataset import OpenBanditDataset, SyntheticBanditDataset, logistic_reward_function
from pydantic import BaseModel
from scipy.stats import rankdata
from sklearn.utils import check_random_state


class RankingBanditFeedback(TypedDict):
    n_rounds: int  # ラウンド数
    n_actions: int  # アクション数
    len_list: int  # ランキングの長さ
    dim_context: int  # 特徴量の次元数
    action_context: np.ndarray  # アクション特徴量 (shape: (n_actions, dim_action_features))
    action: np.ndarray  # 実際に選択されたアクション (shape: (n_rounds * len_list,))
    position: np.ndarray  # ポジション (shape: (n_rounds,))
    reward: np.ndarray  # ポジションレベルの観測報酬 (shape: (n_rounds * len_list)
    pi_b: np.ndarray  # データ収集方策のアクション選択確率 P(a|x) (shape: (n_rounds, n_actions))
    pscore: np.ndarray  # 傾向スコア (shape: (n_rounds,))


class RankingSyntheticBanditDataset(BaseModel):
    """ランキング問題の擬似的なバンディットデータセットを生成するクラス
    実装参考: https://github.com/ghmagazine/cfml_book/blob/main/ch2/dataset.py
    """

    dim_context: int
    num_actions: int
    k: int
    action_context: np.ndarray
    theta: np.ndarray  # 擬似的な期待報酬を生成するための設定値1つ目(d, num_actions)
    M: np.ndarray  # 擬似的な期待報酬を生成するための設定値2つ目 (d, num_actions)
    b: np.ndarray  # 擬似的な期待報酬を生成するための設定値3つ目 (num_actions, 1)
    W: np.ndarray  # ランキング位置間の相互作用を表す重み行列 (k, k)
    beta: float = -1.0  # データ収集方策の設定値1つ目
    reward_noise: float = 0.5  # 観測報酬のばらつき度合い(標準偏差)
    p: list[float] = [0.8, 0.1, 0.1]  # ユーザ行動モデルの選択確率
    p_rand: float = 0.2  # ランダムに選択される確率
    random_state: int = 12345
    is_test: bool = False  # テストモードかどうか

    class Config:
        arbitrary_types_allowed = True  # np.ndarrayを許可

    def obtain_batch_bandit_feedback(self, num_data: int) -> dict:
        random_ = check_random_state(self.random_state)

        # 擬似的な期待報酬関数 q(x, a) を生成
        x, e_a = random_.normal(size=(num_data, self.dim_context)), np.eye(self.num_actions)
        base_q_func = _sigmoid((x**3 + x**2 - x) @ self.theta + (x - x**2) @ self.M @ e_a + self.b.squeeze())

        # ユーザ行動モデルを抽出する
        user_behavior_matrix = np.r_[
            np.eye(self.k),  # independent
            np.tril(np.ones((self.k, self.k))),  # cascade
            np.ones((self.k, self.k)),  # all
        ].reshape((3, self.k, self.k))
        user_behavior_idx = random_.choice(3, p=self.p, size=num_data)
        C_ = user_behavior_matrix[user_behavior_idx]

        user_behavior_matrix_rand = random_.choice([-1, 0, 1], p=[0.2, 0.4, 0.4], size=7 * self.k * self.k).reshape(
            (7, self.k, self.k)
        )
        user_behavior_rand_idx = random_.choice(7, size=num_data)
        C_rand = user_behavior_matrix_rand[user_behavior_rand_idx]

        is_rand = random_.binomial(2, p=self.p_rand, size=num_data).reshape(num_data, 1, 1)
        C = np.clip(C_ + is_rand * C_rand, 0, 1)

        # データ収集方策 pi_0 を定義
        if self.is_test:
            pi_0 = _eps_greedy_policy(base_q_func)
        else:
            pi_0 = _softmax(self.beta * base_q_func)

        # データ収集方策 pi_0 を稼働させた場合のバンディットフィードバックをシミュレーション

        ## シミュレーション結果を格納するための配列を初期化
        a_k = np.zeros((num_data, self.k), dtype=int)
        r_k = np.zeros((num_data, self.k), dtype=float)
        q_k = np.zeros((num_data, self.k), dtype=float)
        ## データ収集方策 pi_0 に従ってアクションをサンプリング
        for k in range(self.k):
            a_k_ = _sample_action_fast(pi_0, random_state=self.random_state + k)
            a_k[:, k] = a_k_
        ## 選ばれたアクションに対して観測される報酬をサンプリング
        idx = np.arange(num_data)
        for k in range(self.k):
            q_func_factual = base_q_func[idx, a_k[:, k]] / self.k
            for l in range(self.k):
                if l != k:
                    q_func_factual += C[:, k, l] * self.W[k, l] * base_q_func[idx, a_k[:, l]] / np.abs(l - k)
            q_k[:, k] = q_func_factual
            r_k[:, k] = random_.normal(q_func_factual, scale=self.reward_noise)

        return dict(
            num_data=num_data,
            K=self.k,
            num_actions=self.num_actions,
            x=x,
            a_k=a_k,
            r_k=r_k,
            C=C,
            pi_0=pi_0,
            q_k=q_k,
            base_q_func=base_q_func,
        )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """シグモイド関数."""
    return np.exp(np.minimum(x, 0)) / (1.0 + np.exp(-np.abs(x)))


def _softmax(x: np.ndarray) -> np.ndarray:
    """ソフトマックス関数."""
    b = np.max(x, axis=1)[:, np.newaxis]
    numerator = np.exp(x - b)
    denominator = np.sum(numerator, axis=1)[:, np.newaxis]
    return numerator / denominator


def _eps_greedy_policy(
    q_func: np.ndarray,
    eps: float = 0.5,
) -> np.ndarray:
    """epsilon-greedy法により方策を定義する."""
    is_topk = rankdata(-q_func, axis=1) <= 1
    pi = (1.0 - eps) * is_topk
    pi += eps / q_func.shape[1]

    return pi


def _sample_action_fast(pi: np.ndarray, random_state: int = 12345) -> np.ndarray:
    """与えられた方策に従い、行動を高速に抽出する."""
    random_ = check_random_state(random_state)
    uniform_rvs = random_.uniform(size=pi.shape[0])[:, np.newaxis]
    cum_pi = pi.cumsum(axis=1)
    flg = cum_pi > uniform_rvs
    sampled_actions = flg.argmax(axis=1)
    return sampled_actions
