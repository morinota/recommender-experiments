import numpy as np
from pandas import DataFrame
from scipy.stats import rankdata


def eps_greedy_policy(
    expected_reward: np.ndarray,
    k: int = 1,
    eps: float = 0.1,
) -> np.ndarray:
    """epsilon-greedy法により方策を定義する."""
    ## 期待報酬が上位k個だったらTrue, それ以外はFalseの配列を作成
    is_topk_vec = rankdata(-expected_reward, axis=1) <= k

    ## 行動選択確率分布を作成
    pi = ((1.0 - eps) / k) * is_topk_vec
    pi += eps / expected_reward.shape[1]
    pi /= pi.sum(1)[:, np.newaxis]

    return pi[:, :, np.newaxis]  # (n_samples, n_actions, 1)


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
