import numpy as np

from recommender_experiments.service.simulator.on_policy_learning_simulator import (
    run_on_policy_learning_single_simulation,
)


class TestOnPolicyLearningSimulator:

    def test_単一設定のシミュレーションが指定された回数だけ実行されること(self):
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
            n_simulations,
            n_actions,
            dim_context,
            action_context,
            n_round_before_deploy,
            n_round_after_deploy,
            logging_policy_function,
        )

        assert len(actual) == n_simulations, "指定回数のシミュレーション結果を返す"
