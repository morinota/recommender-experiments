import collections
import itertools
from dataclasses import dataclass
from typing import Callable, TypedDict

import numpy as np
from obp.dataset import OpenBanditDataset, SyntheticBanditDataset, logistic_reward_function
from scipy.stats import rankdata
from sklearn.utils import check_random_state


class RankingBanditFeedback(TypedDict):
    n_rounds: int  # ラウンド数
    n_actions: int  # アクション数
    dim_context: int  # 特徴量の次元数
    action_context: np.ndarray  # アクション特徴量 (shape: (n_actions, dim_action_features))
    ranking_candidates: list[list[int]]  # ランキング候補の一覧
    action: np.ndarray  # 実際に選択されたアクション (shape: (n_rounds,))
    position: np.ndarray  # ポジション (shape: (n_rounds,))
    reward: np.ndarray  # ポジションレベルの報酬 (shape: (n_rounds * len_list)
    pi_b: np.ndarray  # データ収集方策 P(a|x) (shape: (n_rounds, n_actions))
    pscore: np.ndarray  # 傾向スコア (shape: (n_rounds,))


@dataclass
class RankingSyntheticBanditDataset:
    dim_context: int
    n_actions: int
    len_list: int
    position_level_reward_type: str = "binary"
    position_level_expected_reward_function: Callable = None
    action_context: np.ndarray = None
    behavior_policy_function: Callable = None
    random_state: int = 12345
    dataset_name: str = "ranking_synthetic_bandit_dataset"

    def obtain_batch_bandit_feedback(self, n_rounds: int) -> RankingBanditFeedback:
        """Obtain batch logged bandit data."""
        np.random.seed(self.random_state)

        # コンテキスト生成 (shape: (n_rounds, dim_context))
        contexts = np.random.normal(loc=0, scale=1, size=(n_rounds, self.dim_context))

        # ランキング候補を生成
        ranking_candidates = list(itertools.permutations(range(self.n_actions), self.len_list))
        print(f"{ranking_candidates=}")

        # 期待報酬関数 (position_level_expected_reward_function) が指定されていない場合は、シグモイド内積モデルを使う
        if self.position_level_expected_reward_function is None:
            expected_reward_ = self._default_position_level_reward(contexts)
        position_level_expected_reward_ = self._calc_position_level_expected_reward(contexts)

    def _calc_position_level_expected_reward(self, contexts: np.ndarray) -> np.ndarray:
        """各contextに対して、各アクションのポジションレベルの期待報酬を計算する.
        Args:
            contexts (np.ndarray): 特徴量ベクトル (n_rounds, dim_context)
        Returns:
            np.ndarray: ポジションレベルの期待報酬 (
        """
        n_rounds = contexts.shape[0]

    def _default_position_level_reward(self, contexts: np.ndarray) -> np.ndarray:
        """デフォルトの期待報酬関数:
        - contexts * action_context の内積を計算しsigmoid関数で0-1に変換したものを、
            ポジションレベルの報酬として返す.
        Args:
            contexts (np.ndarray): 特徴量ベクトル (n_rounds, dim_context)
        Returns:
            np.ndarray: ポジションレベルの報酬 (n_rounds, n
        """
        n_rounds = contexts.shape[0]

        # アクション特徴量が指定されていない場合は、ランダムに生成
        if self.action_context is None:
            self.action_context = np.random.normal(loc=0, scale=1, size=(self.n_actions, self.dim_context))

        # ポジションレベルの報酬を計算
        position_level_reward = np.zeros(n_rounds * self.len_list)
        for i in range(n_rounds):
            for k in range(self.len_list):
                position_level_reward[i * self.len_list + k] = np.dot(contexts[i], self.action_context[k])

        return position_level_reward


def generate_synthetic_ranking_feedback(
    num_data: int,
    dim_context: int,
    num_actions: int,
    K: int,
    theta: np.ndarray,  # d * num_actions
    M: np.ndarray,  # d * num_actions
    b: np.ndarray,  # num_actions * 1
    W: np.ndarray,  # K * K
    beta: float = -1.0,
    reward_noise: float = 0.5,
    p: list[float] = [1.0, 0.0, 0.0],  # independent, cascade, all
    p_rand: float = 0.0,  # ランダムに選択される確率
    is_test: bool = False,
    random_state: int = 12345,
) -> dict:
    """ランキングにおけるオフ方策評価におけるログデータを生成する関数.
    Args:
        num_data (int): 生成したいデータ数
        dim_context (int): コンテキストの次元数
        num_actions (int): アクションの数
        K (int): ランキングの長さ
        theta (np.ndarray): 擬似的な期待報酬を生成するための設定値1つ目。
        M (np.ndarray): 擬似的な期待報酬を生成するための設定値2つ目。
        b (np.ndarray): 擬似的な期待報酬を生成するための設定値3つ目。
        W (np.ndarray): ランキング位置間の相互作用を表す重み行列。
        beta (float): データ収集方策の設定値1つ目。
            ソフトマックス関数の温度パラメータ。負の値ほど活用的な方策・0に近いほど探索的な方策。
        reward_noise (float): 観測報酬のばらつき度合い(標準偏差)。
        p (list[float]): ユーザ行動モデルの選択確率。
            [independent, cascade, all]の順で、独立、カスケード、全ての行動を選択する確率。
            ex. [0.8, 0.1, 0.1]ならば、80%の確率で独立、10%の確率でカスケード、10%の確率で全ての行動を選択する。
        p_rand (float):
            ランダムユーザ行動(外れユーザ)の確率。変なユーザ行動パターンを混ぜる用。
        is_test (bool): 評価方策を使うかどうか。
            Trueならばepsilon-greedy方策を使用。Falseならばソフトマックス方策を使用。
        random_state (int): 乱数シード値。再現性担保のために使用。
    Returns:
        dict: 生成されたデータセットの情報を含む辞書。
            - num_data: 生成したデータ数
            - K: ランキングの長さ
            - num_actions: アクションの数
            - x: コンテキストデータ (shape: (num_data, dim_context))
            - a_k: 選択されたアクション (shape: (num_data, K))
            - r_k: 各アクションに対する報酬 (shape: (num_data, K))
            - C: ユーザ行動モデルの行動行列 (shape: (num_data, K, K))
            - pi_0: データ収集方策の各アクションに対する選択確率 (shape: (num_data, num_actions))
            - q_k: 各アクションに対する期待報酬 (shape: (num_data, K))
            - base_q_func: 基本的な期待報酬関数 (shape: (num_data, num_actions))
    """
    random_ = check_random_state(random_state)

    # 擬似的な期待報酬関数 q(x, a) を生成
    x, e_a = random_.normal(size=(num_data, dim_context)), np.eye(num_actions)
    base_q_func = _sigmoid((x**3 + x**2 - x) @ theta + (x - x**2) @ M @ e_a + b.squeeze())

    # ユーザ行動モデルを抽出する
    user_behavior_matrix = np.r_[
        np.eye(K),  # independent
        np.tril(np.ones((K, K))),  # cascade
        np.ones((K, K)),  # all
    ].reshape((3, K, K))
    user_behavior_idx = random_.choice(3, p=p, size=num_data)
    C_ = user_behavior_matrix[user_behavior_idx]

    user_behavior_matrix_rand = random_.choice([-1, 0, 1], p=[0.2, 0.4, 0.4], size=7 * K * K).reshape((7, K, K))
    user_behavior_rand_idx = random_.choice(7, size=num_data)
    C_rand = user_behavior_matrix_rand[user_behavior_rand_idx]

    is_rand = random_.binomial(2, p=p_rand, size=num_data).reshape(num_data, 1, 1)
    C = np.clip(C_ + is_rand * C_rand, 0, 1)

    # データ収集方策 pi_0 を定義
    if is_test:
        pi_0 = _eps_greedy_policy(base_q_func)
    else:
        pi_0 = _softmax(beta * base_q_func)

    # データ収集方策 pi_0 を稼働させた場合のバンディットフィードバックをシミュレーション

    ## シミュレーション結果を格納するための配列を初期化
    a_k = np.zeros((num_data, K), dtype=int)
    r_k = np.zeros((num_data, K), dtype=float)
    q_k = np.zeros((num_data, K), dtype=float)
    ## データ収集方策 pi_0 に従ってアクションをサンプリング
    for k in range(K):
        a_k_ = _sample_action_fast(pi_0, random_state=random_state + k)
        a_k[:, k] = a_k_
    ## 選ばれたアクションに対して観測される報酬をサンプリング
    idx = np.arange(num_data)
    for k in range(K):
        q_func_factual = base_q_func[idx, a_k[:, k]] / K
        for l in range(K):
            if l != k:
                q_func_factual += C[:, k, l] * W[k, l] * base_q_func[idx, a_k[:, l]] / np.abs(l - k)
        q_k[:, k] = q_func_factual
        r_k[:, k] = random_.normal(q_func_factual, scale=reward_noise)

    return dict(
        num_data=num_data,
        K=K,
        num_actions=num_actions,
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


if __name__ == "__main__":
    # 行動空間が5P2=20通りの、ランキングバンディットデータセットを生成
    # dataset = RankingSyntheticBanditDataset(n_actions=5, len_list=2, dim_context=5)
    # bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=1000)
    # print(bandit_feedback)

    print(
        generate_synthetic_ranking_feedback(
            num_data=3,
            dim_context=5,
            num_actions=5,
            K=3,
            theta=np.random.normal(size=(5, 5)),
            M=np.random.normal(size=(5, 5)),
            b=np.random.normal(size=(5, 1)),
            W=np.random.normal(size=(3, 3)),
            beta=-1.0,
            reward_noise=0.5,
            p=[0.8, 0.1, 0.1],
            p_rand=0.2,
            is_test=False,
        )
    )
