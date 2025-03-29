import abc

import numpy as np

from recommender_experiments.service.opl.policy_strategy_interface import (
    PolicyStrategyInterface,
)


class EnvironmentStrategyInterface(abc.ABC):
    """
    環境が実装する共通のインターフェース(Strategy patternにおけるStrategy)
    """

    @abc.abstractmethod
    def create_bandit_feedback(
        self,
        logging_policy_strategy: PolicyStrategyInterface,
        n_rounds: int,
    ) -> np.ndarray:
        """バンディットフィードバックを生成するメソッド
        Args:
            logging_policy_strategy (PolicyStrategyInterface): データ収集方策のstrategy
            n_rounds (int): ラウンド数
        """
        raise NotImplementedError
