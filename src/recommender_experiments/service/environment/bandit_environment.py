"""バンディット環境の基底クラスと実装."""

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
from sklearn.utils import check_random_state

from .ranking_synthetic_dataset import RankingSyntheticBanditDataset


class BanditEnvironmentInterface(ABC):
    """バンディット環境の共通インターフェース."""

    @abstractmethod
    def reset(self) -> None:
        """環境をリセットする."""
        pass

    @abstractmethod
    def get_context_and_available_actions(self, trial: int) -> Tuple[np.ndarray, np.ndarray]:
        """指定トライアルでのコンテキストと利用可能actionを取得する.

        Parameters
        ----------
        trial : int
            試行番号

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (context, available_actions) のタプル
        """
        pass

    @abstractmethod
    def get_rewards(self, context: np.ndarray, selected_actions: List[int], trial: int) -> List[float]:
        """選択された行動に対する報酬を取得する.

        Parameters
        ----------
        context : np.ndarray
            コンテキスト特徴量
        selected_actions : List[int]
            選択された行動のリスト
        trial : int
            試行番号

        Returns
        -------
        List[float]
            各行動に対応する報酬のリスト
        """
        pass

    @abstractmethod
    def get_optimal_reward(self, context: np.ndarray, available_actions: np.ndarray, k: int) -> float:
        """最適報酬を計算する（regret計算用）.

        Parameters
        ----------
        context : np.ndarray
            コンテキスト特徴量
        available_actions : np.ndarray
            利用可能なaction IDの配列
        k : int
            選択する行動数

        Returns
        -------
        float
            最適報酬
        """
        pass


class RankingSyntheticEnvironment(BanditEnvironmentInterface):
    """RankingSyntheticBanditDatasetを環境として使用するクラス.

    既存のRankingSyntheticBanditDatasetを「環境」として利用し、
    オンラインバンディット学習に適した形で提供する。

    Parameters
    ----------
    dataset : RankingSyntheticBanditDataset
        ベースとなる合成データセット
    random_state : int
        乱数シード
    """

    def __init__(self, dataset: RankingSyntheticBanditDataset, random_state: int = 42):
        self.dataset = dataset
        self.random_state = random_state
        self.random_ = check_random_state(random_state)

        # バッチデータを事前生成してキャッシュ（効率化のため）
        self._batch_data = None
        self._batch_size = 0

    def reset(self) -> None:
        """環境をリセットする."""
        self.random_ = check_random_state(self.random_state)
        # バッチデータキャッシュをクリア
        self._batch_data = None
        self._batch_size = 0

    def get_context_and_available_actions(self, trial: int) -> Tuple[np.ndarray, np.ndarray]:
        """指定トライアルでのコンテキストと利用可能actionを取得する.

        Parameters
        ----------
        trial : int
            試行番号

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (context, available_actions) のタプル
        """
        # 決定論的にコンテキストを生成（trial番号をシードとして使用）
        trial_random = check_random_state(self.random_state + trial)
        context = trial_random.normal(size=(self.dataset.dim_context,))

        # 利用可能actionを取得
        available_actions = self._get_available_actions_for_trial(trial)

        return context, available_actions

    def get_rewards(self, context: np.ndarray, selected_actions: List[int], trial: int) -> List[float]:
        """選択された行動に対する報酬を取得する.

        Parameters
        ----------
        context : np.ndarray
            コンテキスト特徴量
        selected_actions : List[int]
            選択された行動のリスト
        trial : int
            試行番号

        Returns
        -------
        List[float]
            各行動に対応する報酬のリスト
        """
        rewards = []

        # 試行固有の乱数を使用（再現性確保）
        reward_random = check_random_state(self.random_state + trial + 10000)

        for k, action_id in enumerate(selected_actions):
            if k < self.dataset.k and action_id < self.dataset.num_actions:
                # 基本Q関数から期待報酬を計算
                expected_reward = self._compute_true_expected_reward(context, action_id)

                # ノイズを追加
                noise = reward_random.normal(0, self.dataset.reward_noise)
                reward = expected_reward + noise
                rewards.append(reward)
            else:
                rewards.append(0.0)

        return rewards

    def get_optimal_reward(self, context: np.ndarray, available_actions: np.ndarray, k: int) -> float:
        """最適報酬を計算する（regret計算用）.

        Parameters
        ----------
        context : np.ndarray
            コンテキスト特徴量
        available_actions : np.ndarray
            利用可能なaction IDの配列
        k : int
            選択する行動数

        Returns
        -------
        float
            最適報酬
        """
        if len(available_actions) == 0:
            return 0.0

        # 利用可能な各actionの期待報酬を計算
        action_expected_rewards = []
        for action_id in available_actions:
            expected_reward = self._compute_true_expected_reward(context, action_id)
            action_expected_rewards.append((expected_reward, action_id))

        # 期待報酬順でソートしてtop-kの報酬を合計
        action_expected_rewards.sort(key=lambda x: x[0], reverse=True)
        optimal_reward = sum([reward for reward, _ in action_expected_rewards[:k]])

        return optimal_reward

    def _get_available_actions_for_trial(self, trial: int) -> np.ndarray:
        """指定トライアルで利用可能なactionを取得する.

        Parameters
        ----------
        trial : int
            試行番号

        Returns
        -------
        np.ndarray
            利用可能なaction IDの配列
        """
        if self.dataset.action_churn_schedule is not None:
            # スケジュールに基づいて利用可能actionを決定
            applicable_actions = None
            for start_idx in sorted(self.dataset.action_churn_schedule.keys(), reverse=True):
                if trial >= start_idx:
                    applicable_actions = self.dataset.action_churn_schedule[start_idx]
                    break

            if applicable_actions is not None:
                return np.array(applicable_actions)

        # デフォルトでは全action利用可能
        return np.arange(self.dataset.num_actions)

    def _compute_true_expected_reward(self, context: np.ndarray, action_id: int) -> float:
        """指定したcontextとactionに対する真の期待報酬を計算する.

        Parameters
        ----------
        context : np.ndarray
            コンテキスト特徴量
        action_id : int
            action ID

        Returns
        -------
        float
            真の期待報酬
        """
        # RankingSyntheticBanditDatasetの報酬生成ロジックと同じ計算
        x_transformed = context**3 + context**2 - context
        linear_term = np.dot(x_transformed, self.dataset.theta[:, action_id])

        e_a = np.zeros(self.dataset.num_actions)
        e_a[action_id] = 1
        x_quad_transformed = context - context**2
        quadratic_term = np.dot(x_quad_transformed, np.dot(self.dataset.quadratic_weights, e_a))

        sigmoid_input = linear_term + quadratic_term + self.dataset.action_bias[action_id, 0]
        expected_reward = 1.0 / (1.0 + np.exp(-sigmoid_input))  # sigmoid

        return expected_reward
