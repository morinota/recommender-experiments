from typing import Callable, Literal

import numpy as np
import pydantic

from recommender_experiments.service.environment.environment_strategy_interface import EnvironmentStrategyInterface
from recommender_experiments.service.opl.policy_strategy_interface import PolicyStrategyInterface


class OnPolicyLearningSimulationResult(pydantic.BaseModel):
    simulation_idx: int
    n_actions: int
    dim_context: int
    n_rounds_before_deploy: int
    n_rounds_after_deploy: int
    expected_reward_name: str
    off_policy_learning_method: Literal["ips", "dr", "regression_based"]
    n_epochs: int
    learning_rate_init: float
    batch_size: int
    logging_policy_value: float
    new_policy_value_before_deploy: float
    new_policy_value_after_deploy: float


def run_on_policy_learning_single_simulation(
    target_policy_strategy: PolicyStrategyInterface,
    n_simulations: int,
    n_rounds_before_deploy: int,
    n_rounds_after_deploy: int,
    logging_policy_strategy: PolicyStrategyInterface,
    environment_strategy: EnvironmentStrategyInterface,
    off_policy_learning_method: str = "dr",
    n_epochs: int = 200,
    batch_size: int = 32,
    learning_rate_init: float = 0.00001,
) -> list[OnPolicyLearningSimulationResult]:
    # 真の方策性能の評価用のbandit_feedbackを用意しておく
    bandit_feedback_test = environment_strategy.obtain_batch_bandit_feedback(
        logging_policy_strategy=logging_policy_strategy,  # ここはなんでもいい
        n_rounds=10000,
    )

    results = []
    for simulation_idx in range(n_simulations):
        # データ収集方策を学習
        bandit_feedback_before_deploy = environment_strategy.obtain_batch_bandit_feedback(
            logging_policy_strategy=logging_policy_strategy, n_rounds=n_rounds_before_deploy
        )
        # データ収集方策の方策性能を評価
        logging_policy_value = environment_strategy.calc_policy_value(
            expected_reward=bandit_feedback_test.expected_reward,
            action_dist=logging_policy_strategy.predict_proba(
                context=bandit_feedback_test.context, action_context=bandit_feedback_test.action_context
            ),
        )

        # 新方策をオフ方策学習
        target_policy_strategy.fit(
            bandit_feedback_train=bandit_feedback_before_deploy.model_dump(),
            bandit_feedback_test=bandit_feedback_test.model_dump(),
        )
        # 新方策のデプロイ前の方策性能を評価
        new_policy_value_before_deploy = environment_strategy.calc_policy_value(
            expected_reward=bandit_feedback_test.expected_reward,
            action_dist=target_policy_strategy.predict_proba(
                context=bandit_feedback_test.context, action_context=bandit_feedback_test.action_context
            ),
        )

        # 新方策をデプロイして新たにbandit_feedbackを取得
        bandit_feedback_after_deploy = environment_strategy.obtain_batch_bandit_feedback(
            logging_policy_strategy=target_policy_strategy, n_rounds=n_rounds_after_deploy
        )
        # 新方策をオンライン学習
        target_policy_strategy.fit(
            bandit_feedback_train=bandit_feedback_after_deploy.model_dump(),
            bandit_feedback_test=bandit_feedback_test.model_dump(),
        )
        # 新方策のデプロイ後の方策性能を評価
        new_policy_value_after_deploy = environment_strategy.calc_policy_value(
            expected_reward=bandit_feedback_test.expected_reward,
            action_dist=target_policy_strategy.predict_proba(
                context=bandit_feedback_test.context, action_context=bandit_feedback_test.action_context
            ),
        )

        results.append(
            OnPolicyLearningSimulationResult(
                simulation_idx=simulation_idx,
                n_actions=environment_strategy.n_actions,
                dim_context=environment_strategy.dim_context,
                n_rounds_before_deploy=n_rounds_before_deploy,
                n_rounds_after_deploy=n_rounds_before_deploy,
                off_policy_learning_method=off_policy_learning_method,
                n_epochs=n_epochs,
                expected_reward_name=environment_strategy.expected_reward_strategy_name,
                learning_rate_init=learning_rate_init,
                batch_size=batch_size,
                logging_policy_value=logging_policy_value,
                new_policy_value_before_deploy=new_policy_value_before_deploy,
                new_policy_value_after_deploy=new_policy_value_after_deploy,
            )
        )

    return results
