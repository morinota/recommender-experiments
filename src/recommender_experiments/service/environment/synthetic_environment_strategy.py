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


class SyntheticEnvironmentStrategy(EnvironmentStrategyInterface):

    def __init__(
        self,
        n_actions: int = 5,
        dim_context: int = 4,
        action_context: np.ndarray = None,
    ):
        self.__n_actions = n_actions
        self.__dim_context = dim_context
        self.__action_context = action_context

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
        )

        bandit_feedback_dict = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
        return BanditFeedbackModel(**bandit_feedback_dict)

    def calc_ground_truth_policy_value(
        self,
        expected_reward: np.ndarray,
        action_dist: np.ndarray,
    ) -> float:
        # V(π)の定義は $:= E_{p(x, a ,r)}[r] = E_{p(x) \pi(a|x) p(r|x,a)}[r]$ とする
        policy_value = np.average(
            expected_reward, weights=action_dist[:, :, 0], axis=1
        ).mean()
        return policy_value
