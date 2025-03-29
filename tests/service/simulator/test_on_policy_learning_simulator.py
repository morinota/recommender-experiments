import numpy as np

from recommender_experiments.service.opl.policy_strategy_interface import (
    PolicyStrategyInterface,
)
from recommender_experiments.service.simulator.on_policy_learning_simulator import (
    OnPolicyLearningSimulationResult,
    run_on_policy_learning_single_simulation,
)


class DummyPolicyStrategy(PolicyStrategyInterface):
    def __init__(self):
        pass

    def fit(self) -> None:
        pass

    def predict_proba(
        self, context: np.ndarray, action_context: np.ndarray
    ) -> np.ndarray:
        n_rounds = context.shape[0]
        # dummryなので、contextに依らず全てのroundで一様な行動選択確率を返す!
        return np.full((n_rounds, self.n_actions), 1.0 / self.n_actions)

    @property
    def policy_name(self) -> str:
        return "DummyPolicyStrategy"


def test_単一設定のシミュレーションが指定された回数だけ実行されること():
    # Arrange
    n_simulations = 2
    n_actions = 10
    dim_context = 5
    action_context = np.random.random(size=(n_actions, dim_context))
    n_round_before_deploy = 1000
    n_round_after_deploy = 1000

    # データ収集方策は簡単のため、一様ランダムな方策を指定
    logging_policy_function = lambda context, action_context, random_state: np.full(
        (context.shape[0], n_actions), 1.0 / n_actions
    )
    # 真の期待報酬 E_{p(r|x,a)}[r] の設定
    expected_reward_lower = 0.0
    expected_reward_upper = 0.5
    expected_reward_setting = "my_context_free"

    # Act
    actual = run_on_policy_learning_single_simulation(
        policy_strategy=DummyPolicyStrategy(),
        n_simulations=n_simulations,
        n_actions=n_actions,
        dim_context=dim_context,
        action_context=action_context,
        n_rounds_before_deploy=n_round_before_deploy,
        n_rounds_after_deploy=n_round_after_deploy,
        logging_policy_function=logging_policy_function,
    )

    # Assert
    assert len(actual) == n_simulations, "指定回数のシミュレーション結果を返す"
    assert all(
        isinstance(result, OnPolicyLearningSimulationResult) for result in actual
    ), "OPLSimulationResultのリストを返す"
