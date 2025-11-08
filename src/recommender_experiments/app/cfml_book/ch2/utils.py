import numpy as np
from pandas import DataFrame
from scipy.stats import rankdata

from recommender_experiments.app.cfml_book.common_utils import sample_action_fast, sigmoid, softmax


def eps_greedy_policy(
    q_func: np.ndarray,
    eps: float = 0.5,
) -> np.ndarray:
    """epsilon-greedy法により方策を定義する."""
    is_topk = rankdata(-q_func, axis=1) <= 1
    pi = (1.0 - eps) * is_topk
    pi += eps / q_func.shape[1]

    return pi


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
