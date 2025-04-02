import numpy as np

from recommender_experiments.service.simulator.opl_simulator import (
    OPLSimulationResult,
    run_opl_multiple_simulations_in_parallel,
    run_opl_single_simulation,
)


class TestOPLSimulator:
    def test_単一設定のOPLシミュレーションが指定された回数だけ実行されること(self):
        # Arrange
        n_simulations = 2
        n_actions = 10
        dim_context = 5
        n_rounds_train = 100
        n_rounds_test = 100
        batch_size = 32
        n_epochs = 2
        action_context = np.random.random(size=(n_actions, dim_context))
        # データ収集方策は簡単のため、一様ランダムな方策を指定
        logging_policy_function = lambda context, action_context, random_state: np.full(
            (context.shape[0], n_actions), 1.0 / n_actions
        )
        # 真の期待報酬 E_{p(r|x,a)}[r] の設定
        expected_reward_lower = 0.0
        expected_reward_upper = 0.5
        expected_reward_setting = "my_context_free"
        # 新方策の設定
        new_policy_setting = "two_tower_nn"

        # Act
        actual = run_opl_single_simulation(
            n_simulations=n_simulations,
            n_actions=n_actions,
            dim_context=dim_context,
            action_context=action_context,
            n_rounds_train=n_rounds_train,
            n_rounds_test=n_rounds_test,
            batch_size=batch_size,
            n_epochs=n_epochs,
            logging_policy_function=logging_policy_function,
            expected_reward_setting=expected_reward_setting,
            expected_reward_lower=expected_reward_lower,
            expected_reward_upper=expected_reward_upper,
            new_policy_setting=new_policy_setting,
        )

        # Assert
        assert len(actual) == n_simulations, "指定回数のシミュレーション結果を返す"
        assert all(isinstance(result, OPLSimulationResult) for result in actual), "OPLSimulationResultのリストを返す"

    def test_複数設定のOPLシミュレーションが並列で実行されること(self):
        # Arrange
        n_simulations = 2
        n_actions_list = [5, 10]
        dim_context_list = [5, 10]
        n_rounds_train_list = [50, 100]
        n_rounds_test_list = [50, 100]
        batch_size_list = [32, 64]
        n_epochs_list = [2, 4]
        expected_reward_scale_list = [(0.0, 0.1), (0.0, 0.5)]
        expected_reward_settings = ["my_context_free", "my_context_aware"]
        new_policy_settings = ["two_tower_nn", "obp_nn"]
        logging_policy_functions = [
            # データ収集方策は簡単のため、一様ランダムな方策のみ指定
            lambda context, action_context, random_state: np.full(
                (context.shape[0], action_context.shape[0]), 1.0 / action_context.shape[0]
            )
        ]
        n_jobs = 2

        # Act
        actual = run_opl_multiple_simulations_in_parallel(
            n_simulations,
            n_actions_list,
            dim_context_list,
            n_rounds_train_list,
            n_rounds_test_list,
            batch_size_list,
            n_epochs_list,
            expected_reward_scale_list,
            expected_reward_settings,
            new_policy_settings,
            n_jobs,
            logging_policy_functions,
        )

        # Assert
        assert len(actual) == (
            n_simulations
            * len(n_actions_list)
            * len(dim_context_list)
            * len(n_rounds_train_list)
            * len(n_rounds_test_list)
            * len(batch_size_list)
            * len(n_epochs_list)
            * len(expected_reward_scale_list)
            * len(expected_reward_settings)
            * len(new_policy_settings)
            * len(logging_policy_functions)
        ), "「指定回数 * 各設定の組み合わせ総数」のレコード数の結果を返す"
        assert all(isinstance(result, OPLSimulationResult) for result in actual), "OPLSimulationResultのリストを返す"
