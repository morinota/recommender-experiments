from typing import Callable, Literal
import numpy as np
import pydantic

from recommender_experiments.service.opl.policy_strategy_interface import (
    PolicyStrategyInterface,
)
from obp.dataset import SyntheticBanditDataset, logistic_reward_function


class OnPolicyLearningSimulationResult(pydantic.BaseModel):
    simulation_idx: int
    n_actions: int
    dim_context: int
    n_rounds_before_deploy: int
    n_rounds_aflter_deploy: int
    expected_reward_lower: float
    expected_reward_upper: float
    expected_reward_setting: Literal["my_context_free", "my_context_aware", "linear"]
    off_policy_learning_method: Literal["ips", "dr", "regression_based"]
    n_epochs: int
    learning_rate_init: float
    batch_size: int
    logging_policy_value: float
    new_policy_value_before_deploy: float
    new_policy_value_after_deploy: float


def run_on_policy_learning_single_simulation(
    policy_strategy: PolicyStrategyInterface,
    n_simulations: int,
    n_actions: int,
    dim_context: int,
    action_context: np.ndarray,
    n_rounds_before_deploy: int,
    n_rounds_after_deploy: int,
    logging_policy_function: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
    expected_reward_lower: float = 0.0,
    expected_reward_upper: float = 0.5,
    expected_reward_setting: Literal[
        "my_context_free", "my_context_aware", "linear"
    ] = "my_context_aware",
    off_policy_learning_method: str = "dr",
    n_epochs: int = 200,
    batch_size: int = 32,
    learning_rate_init: float = 0.00001,
) -> list[OnPolicyLearningSimulationResult]:
    results = []
    for simulation_idx in range(n_simulations):
        dataset = SyntheticBanditDataset(
            n_actions=n_actions,
            dim_context=dim_context,
            reward_type="binary",
            reward_function=logistic_reward_function,
            behavior_policy_function=logging_policy_function,
            random_state=simulation_idx,
            action_context=action_context,
        )

        logging_policy_value = 1.0
        new_policy_value_before_deploy = 1.0
        new_policy_value_after_deploy = 1.0
        results.append(
            OnPolicyLearningSimulationResult(
                simulation_idx=simulation_idx,
                n_actions=n_actions,
                dim_context=dim_context,
                n_rounds_before_deploy=n_rounds_before_deploy,
                n_rounds_aflter_deploy=n_rounds_before_deploy,
                expected_reward_lower=expected_reward_lower,
                expected_reward_upper=expected_reward_upper,
                expected_reward_setting=expected_reward_setting,
                off_policy_learning_method=off_policy_learning_method,
                n_epochs=n_epochs,
                learning_rate_init=learning_rate_init,
                batch_size=batch_size,
                logging_policy_value=logging_policy_value,
                new_policy_value_before_deploy=new_policy_value_before_deploy,
                new_policy_value_after_deploy=new_policy_value_after_deploy,
            )
        )

    return results
