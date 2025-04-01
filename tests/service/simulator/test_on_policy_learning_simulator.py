import numpy as np

from recommender_experiments.service.environment.synthetic_environment_strategy import (
    SyntheticEnvironmentStrategy,
)
from recommender_experiments.service.opl.policy_strategy_interface import (
    PolicyStrategyInterface,
)
from recommender_experiments.service.simulator.on_policy_learning_simulator import (
    OnPolicyLearningSimulationResult,
    run_on_policy_learning_single_simulation,
)


class DummyPolicyStrategy(PolicyStrategyInterface):
    def fit(self, *args, **kwargs) -> None:
        pass

    def predict_proba(
        self, context: np.ndarray, action_context: np.ndarray, random_state: int = 0
    ) -> np.ndarray:
        n_rounds = context.shape[0]
        n_actions = action_context.shape[0]
        # dummryなので、contextに依らず全てのroundで一様な行動選択確率を返す!
        action_dist = np.full((n_rounds, n_actions), 1.0 / n_actions)
        return action_dist

    @property
    def policy_name(self) -> str:
        return "DummyPolicyStrategy"


def test_単一設定のシミュレーションが指定された回数だけ実行されること():
    # Arrange
    n_simulations = 2
    environment_strategy = SyntheticEnvironmentStrategy(
        n_actions=10,
        dim_context=5,
        action_context=np.random.random(size=(10, 5)),
        expected_reward_strategy=None,
    )
    n_round_before_deploy = 1000
    n_round_after_deploy = 1000

    # Act
    actual = run_on_policy_learning_single_simulation(
        n_simulations=n_simulations,
        target_policy_strategy=DummyPolicyStrategy(),
        logging_policy_strategy=DummyPolicyStrategy(),
        environment_strategy=environment_strategy,
        n_rounds_before_deploy=n_round_before_deploy,
        n_rounds_after_deploy=n_round_after_deploy,
    )

    # Assert
    assert all(
        isinstance(result, OnPolicyLearningSimulationResult) for result in actual
    ), "OPLSimulationResultのリストを返す"
    assert len(actual) == n_simulations, "指定回数のシミュレーション結果を返す"
