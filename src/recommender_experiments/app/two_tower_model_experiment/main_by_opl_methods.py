from pathlib import Path
from recommender_experiments.service.simulator.opl_simulator import run_opl_multiple_simulations_in_parallel
import recommender_experiments.service.utils.logging_policies as logging_policies
import polars as pl

RESULT_DIR = Path("logs/two_tower_model_experiment")


def 一様ランダムなデータ収集方策ver() -> None:
    results = run_opl_multiple_simulations_in_parallel(
        n_simulations=5,
        n_actions_list=[10],
        dim_context_list=[50],
        n_rounds_train_list=[2000, 5000, 10000, 15000, 20000],
        n_rounds_test_list=[2000],
        batch_size_list=[10],
        n_epochs_list=[200],
        expected_reward_scale_list=[(0.0, 0.5)],
        expected_reward_settings=["linear"],
        new_policy_settings=["two_tower_nn"],
        logging_policy_functions=[logging_policies.random_policy],
        off_policy_learning_methods=["dr", "ips", "regression_based"],
        n_jobs=5,
    )

    result_df = pl.DataFrame([result.model_dump() for result in results])
    print(result_df)

    # 結果の保存
    result_df.write_csv(RESULT_DIR / "random_logging_policy_results_batch10.csv")


def パーソナライズされたデータ収集方策ver() -> None:
    results = run_opl_multiple_simulations_in_parallel(
        n_simulations=5,
        n_actions_list=[10],
        dim_context_list=[50],
        n_rounds_train_list=[2000, 5000, 10000, 15000, 20000],
        n_rounds_test_list=[2000],
        batch_size_list=[10],
        n_epochs_list=[200],
        expected_reward_scale_list=[(0.0, 0.5)],
        expected_reward_settings=["linear"],
        new_policy_settings=["two_tower_nn"],
        logging_policy_functions=[logging_policies.context_aware_stochastic_policy],
        off_policy_learning_methods=["dr", "ips", "regression_based"],
        n_jobs=5,
    )

    result_df = pl.DataFrame([result.model_dump() for result in results])
    print(result_df)

    # 結果の保存
    result_df.write_csv(RESULT_DIR / "personalized_logging_policy_results_batch10.csv")


def 行動数の大きさがオフライン学習に与える影響() -> None:
    results = run_opl_multiple_simulations_in_parallel(
        n_simulations=5,
        n_actions_list=[10, 50, 100, 200, 500, 1000, 2000, 5000],
        dim_context_list=[50],
        n_rounds_train_list=[15000],
        n_rounds_test_list=[2000],
        batch_size_list=[10],
        n_epochs_list=[200],
        expected_reward_scale_list=[(0.0, 0.5)],
        expected_reward_settings=["linear"],
        new_policy_settings=["two_tower_nn"],
        logging_policy_functions=[logging_policies.context_aware_stochastic_policy],
        off_policy_learning_methods=["dr", "ips", "regression_based"],
        n_jobs=5,
    )

    result_df = pl.DataFrame([result.model_dump() for result in results])
    print(result_df)

    # 結果の保存
    result_df.write_csv(RESULT_DIR / "n_actions_results_batch10.csv")


def バッチサイズの大きさがオフライン学習に与える影響() -> None:
    results = run_opl_multiple_simulations_in_parallel(
        n_simulations=5,
        n_actions_list=[10],
        dim_context_list=[50],
        n_rounds_train_list=[15000],
        n_rounds_test_list=[2000],
        batch_size_list=[10, 100, 200, 500, 1000, 2000, 5000],
        n_epochs_list=[200],
        expected_reward_scale_list=[(0.0, 0.5)],
        expected_reward_settings=["linear"],
        new_policy_settings=["two_tower_nn"],
        logging_policy_functions=[logging_policies.context_aware_stochastic_policy],
        off_policy_learning_methods=["dr", "ips", "regression_based"],
        n_jobs=5,
    )

    result_df = pl.DataFrame([result.model_dump() for result in results])
    print(result_df)

    # 結果の保存
    result_df.write_csv(RESULT_DIR / "batch_size_results.csv")


def main() -> None:
    # 一様ランダムなデータ収集方策ver()
    print("-----------------------------------")
    パーソナライズされたデータ収集方策ver()
    行動数の大きさがオフライン学習に与える影響()
    # バッチサイズの大きさがオフライン学習に与える影響()


if __name__ == "__main__":
    main()
