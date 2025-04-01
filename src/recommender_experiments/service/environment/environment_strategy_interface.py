import abc

import numpy as np

from recommender_experiments.service.opl.policy_strategy_interface import (
    PolicyStrategyInterface,
)
from recommender_experiments.service.synthetic_bandit_feedback import (
    BanditFeedbackModel,
)


class EnvironmentStrategyInterface(abc.ABC):
    """
    環境が実装する共通のインターフェース(Strategy patternにおけるStrategy)
    """

    @property
    @abc.abstractmethod
    def n_actions(self) -> int:
        """アクション数"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def dim_context(self) -> int:
        """コンテキストの次元数"""
        raise NotImplementedError

    @abc.abstractmethod
    def obtain_batch_bandit_feedback(
        self,
        logging_policy_strategy: PolicyStrategyInterface,
        n_rounds: int,
    ) -> BanditFeedbackModel:
        """バンディットフィードバックを生成するメソッド
        Args:
            logging_policy_strategy (PolicyStrategyInterface): データ収集方策のstrategy
            n_rounds (int): ラウンド数
        """
        raise NotImplementedError

    @abc.abstractmethod
    def calc_policy_value(
        self,
        expected_reward: np.ndarray,
        action_dist: np.ndarray,
    ) -> float:
        """真の方策の価値を計算するメソッド
        Args:
            expected_reward (np.ndarray): 期待報酬関数 q(x,a) = E_{p(r|x,a)}[r] の配列 (n_rounds, n_actions)
            action_dist (np.ndarray): 任意の方策 \pi による行動選択確率分布の配列 (n_rounds, n_actions)
        Returns:
            float: 方策性能 V_{\pi}
        """
        raise NotImplementedError
