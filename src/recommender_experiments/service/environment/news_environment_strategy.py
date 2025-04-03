import numpy as np
from recommender_experiments.service.environment.environment_strategy_interface import (
    EnvironmentStrategyInterface,
)
from recommender_experiments.service.synthetic_bandit_feedback import (
    BanditFeedbackModel,
)


class NewsEnvironmentStrategy(EnvironmentStrategyInterface):
    def __init__(self):
        pass

    @property
    def n_actions(self) -> int:
        return 10

    @property
    def dim_context(self) -> int:
        return 100

    @property
    def expected_reward_strategy_name(self) -> str:
        return "実際のデータなので、期待報酬関数は不明"

    def obtain_batch_bandit_feedback(self, n_rounds: int) -> BanditFeedbackModel:
        pass

    def calc_policy_value(
        self,
        expected_reward: np.ndarray,
        action_dist: np.ndarray,
    ) -> float:
        raise Exception("実際のデータなので、真の方策性能は計算できない")
