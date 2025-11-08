import numpy as np
from pandas import DataFrame

from recommender_experiments.app.cfml_book.common_utils import eps_greedy_policy as _eps_greedy_policy


def eps_greedy_policy(
    expected_reward: np.ndarray,
    k: int = 1,
    eps: float = 0.1,
) -> np.ndarray:
    """epsilon-greedy法により方策を定義する."""
    return _eps_greedy_policy(
        expected_reward, k=k, eps=eps, return_normalized=True, rank_method=None, add_newaxis=True
    )


def aggregate_simulation_results(
    estimated_policy_value_list: list,
    policy_value: float,
    experiment_config_name: str,
    experiment_config_value: int,
) -> DataFrame:
    """各推定量の推定値から平均二乗誤差や二乗バイアス、バリアンスなどの実験結果を集計する."""
    result_df = (
        DataFrame(DataFrame(estimated_policy_value_list).stack())
        .reset_index(1)
        .rename(columns={"level_1": "est", 0: "value"})
    )
    result_df[experiment_config_name] = experiment_config_value
    result_df["se"] = (result_df.value - policy_value) ** 2
    result_df["bias"] = 0
    result_df["variance"] = 0
    result_df["true_value"] = policy_value
    sample_mean = DataFrame(result_df.groupby(["est"]).mean().value).reset_index()
    for est_ in sample_mean["est"]:
        estimates = result_df.loc[result_df["est"] == est_, "value"].values
        mean_estimates = sample_mean.loc[sample_mean["est"] == est_, "value"].values
        mean_estimates = np.ones_like(estimates) * mean_estimates
        result_df.loc[result_df["est"] == est_, "bias"] = (policy_value - mean_estimates) ** 2
        result_df.loc[result_df["est"] == est_, "variance"] = (estimates - mean_estimates) ** 2

    return result_df


def aggregate_simulation_results_lam(
    estimated_policy_value_list: list,
    policy_value: float,
    experiment_config_name: str,
    experiment_config_value: int,
) -> DataFrame:
    """各推定量の推定値から平均二乗誤差や二乗バイアス、バリアンスなどの実験結果を集計する."""
    result_df = (
        DataFrame(DataFrame(estimated_policy_value_list).stack())
        .reset_index(1)
        .rename(columns={"level_1": "lam", 0: "value"})
    )
    result_df[experiment_config_name] = experiment_config_value
    result_df["se"] = (result_df.value - policy_value) ** 2
    result_df["bias"] = 0
    result_df["variance"] = 0
    result_df["true_value"] = policy_value
    sample_mean = DataFrame(result_df.groupby(["lam"]).mean().value).reset_index()
    for est_ in sample_mean["lam"]:
        estimates = result_df.loc[result_df["lam"] == est_, "value"].values
        mean_estimates = sample_mean.loc[sample_mean["lam"] == est_, "value"].values
        mean_estimates = np.ones_like(estimates) * mean_estimates
        result_df.loc[result_df["lam"] == est_, "bias"] = (policy_value - mean_estimates) ** 2
        result_df.loc[result_df["lam"] == est_, "variance"] = (estimates - mean_estimates) ** 2

    return result_df
