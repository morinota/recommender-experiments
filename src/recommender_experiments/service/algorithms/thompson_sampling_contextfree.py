"""Context-free Thompson Samplingを使ったランキングバンディットアルゴリズム."""

import numpy as np
from sklearn.utils import check_random_state

from recommender_experiments.service.algorithms.bandit_algorithm_interface import BanditAlgorithmInterface


class ThompsonSamplingContextFree(BanditAlgorithmInterface):
    """Context-free Thompson Samplingを使ったランキングバンディット.

    コンテキスト情報を使用せず、各actionの報酬をBeta分布で近似。
    各actionに対してBeta分布のパラメータ（成功回数・失敗回数）を
    保持し、サンプリングによって行動を選択する。

    Parameters
    ----------
    num_actions : int
        行動数
    k : int
        ランキング長（選択する行動数）
    alpha : float
        事前分布のBetaパラメータ（成功の擬似カウント）
    beta : float
        事前分布のBetaパラメータ（失敗の擬似カウント）
    random_state : int
        乱数シード
    """

    def __init__(
        self,
        num_actions: int,
        k: int,
        alpha: float = 1.0,
        beta: float = 1.0,
        random_state: int = 42,
    ):
        self.num_actions = num_actions
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.random_state = random_state
        self.random_ = check_random_state(random_state)

        # Beta分布のパラメータを初期化
        self.reset()

    def reset(self) -> None:
        """アルゴリズムの状態をリセットする."""
        # 各actionに対するBeta分布のパラメータ
        # successes[a] = alpha + 観測された報酬の総和
        # failures[a] = beta + 観測された試行回数 - 観測された報酬の総和
        self.successes = np.full(self.num_actions, self.alpha)
        self.failures = np.full(self.num_actions, self.beta)

    def select_actions(self, context: np.ndarray, available_actions: np.ndarray, k: int) -> list[int]:
        """利用可能なactionの中からk個の行動を選択する.

        Context-free Thompson Samplingにより、各actionの報酬期待値を
        Beta分布からサンプリングし、上位k個を選択する。
        contextは使用されない。

        Parameters
        ----------
        context : np.ndarray
            コンテキスト特徴量（この実装では使用されない）
        available_actions : np.ndarray
            利用可能なaction IDの配列
        k : int
            選択する行動数

        Returns
        -------
        list[int]
            選択された行動のリスト（長さk）
        """
        if len(available_actions) == 0:
            return []

        # 利用可能な行動数がk未満の場合は全て選択
        k = min(k, len(available_actions))

        sampled_rewards = []

        for action_id in available_actions:
            # Beta分布から期待報酬をサンプリング
            # theta ~ Beta(successes[a], failures[a])
            sampled_reward = self.random_.beta(self.successes[action_id], self.failures[action_id])
            sampled_rewards.append((sampled_reward, action_id))

        # 報酬期待値の降順でソートしてtop-kを選択
        sampled_rewards.sort(key=lambda x: x[0], reverse=True)
        selected_actions = [action_id for _, action_id in sampled_rewards[:k]]

        return selected_actions

    def update(self, context: np.ndarray, selected_actions: list[int], rewards: list[float]) -> None:
        """観測された報酬をもとにBeta分布のパラメータを更新する.

        Parameters
        ----------
        context : np.ndarray
            コンテキスト特徴量（この実装では使用されない）
        selected_actions : list[int]
            選択された行動のリスト
        rewards : list[float]
            各行動に対応する報酬のリスト
        """
        for action_id, reward in zip(selected_actions, rewards):
            if action_id >= self.num_actions:
                continue  # 無効なaction IDはスキップ

            # 報酬を [0, 1] の範囲にクリップ（Beta分布への適合のため）
            reward = np.clip(reward, 0.0, 1.0)

            # Beta分布のパラメータを更新
            # 成功数に報酬を、失敗数に(1-報酬)を加算
            self.successes[action_id] += reward
            self.failures[action_id] += 1.0 - reward

    @property
    def algorithm_name(self) -> str:
        """アルゴリズム名を返す."""
        return f"ThompsonSamplingContextFree(alpha={self.alpha}, beta={self.beta})"

    def get_action_statistics(self) -> dict:
        """各actionの統計情報を取得する（デバッグ用）.

        Returns
        -------
        dict
            action_id -> (successes, failures, expected_reward) の辞書
        """
        stats = {}

        for action_id in range(self.num_actions):
            successes = self.successes[action_id]
            failures = self.failures[action_id]
            expected_reward = successes / (successes + failures)

            stats[action_id] = (successes, failures, expected_reward)

        return stats
