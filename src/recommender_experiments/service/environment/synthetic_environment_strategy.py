import numpy as np
from recommender_experiments.service.environment.environment_strategy_interface import (
    EnvironmentStrategyInterface,
)
from recommender_experiments.service.opl.policy_strategy_interface import (
    PolicyStrategyInterface,
)
from recommender_experiments.service.synthetic_bandit_feedback import (
    BanditFeedbackModel,
)
from obp.dataset import SyntheticBanditDataset

from recommender_experiments.service.utils.expected_reward_functions import (
    ExpectedRewardStrategy,
)


class SyntheticEnvironmentStrategy(EnvironmentStrategyInterface):

    def __init__(
        self,
        n_actions: int = 5,
        dim_context: int = 4,
        action_context: np.ndarray = None,
        expected_reward_strategy: ExpectedRewardStrategy = None,
    ):
        self.__n_actions = n_actions
        self.__dim_context = dim_context
        self.__action_context = action_context
        self.__expected_reward_strategy = expected_reward_strategy
        self.__expected_reward_function = (
            expected_reward_strategy.get_function()
            if expected_reward_strategy
            else None
        )

    @property
    def n_actions(self) -> int:
        return self.__n_actions

    @property
    def dim_context(self) -> int:
        return self.__dim_context

    @property
    def expected_reward_strategy_name(self) -> ExpectedRewardStrategy:
        """期待報酬関数の名前を取得するプロパティ"""
        if self.__expected_reward_strategy is None:
            return "default expected reward function"
        return self.__expected_reward_strategy.name

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

        dataset = SyntheticBanditDataset(
            n_actions=self.__n_actions,
            dim_context=self.__dim_context,
            reward_type="binary",
            behavior_policy_function=logging_policy_strategy.predict_proba,
            action_context=self.__action_context,
            reward_function=self.__expected_reward_function,
        )

        bandit_feedback_dict = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
        return BanditFeedbackModel(**bandit_feedback_dict)

    def calc_policy_value(
        self,
        expected_reward: np.ndarray,
        action_dist: np.ndarray,  # (ラウンド数, アクション数)
    ) -> float:
        if action_dist.ndim == 2:
            # 2次元の場合は、次元を追加して3次元にする
            action_dist = action_dist[:, :, np.newaxis]
        # V(π)の定義は $:= E_{p(x, a ,r)}[r] = E_{p(x) \pi(a|x) p(r|x,a)}[r]$ とする
        policy_value = np.average(
            expected_reward, weights=action_dist[:, :, 0], axis=1
        ).mean()
        return policy_value
