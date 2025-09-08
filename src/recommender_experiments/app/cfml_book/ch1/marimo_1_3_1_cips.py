import marimo

__generated_with = "0.9.24"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    mo.md(r"""# 1.3.1 Clipped Inverse Propensity Score(CIPS)推定量
    参考文献
    - Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudík. [Doubly Robust Off-Policy Evaluation with Shrinkage](https://arxiv.org/abs/1907.09623). ICML2021.""")
    return


@app.cell
def __():
    import warnings

    warnings.filterwarnings("ignore")

    import japanize_matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from pandas import DataFrame
    from sklearn.utils import check_random_state
    from tqdm import tqdm

    plt.style.use("ggplot")
    linestyle_dict = {"se": "-", "bias": "--", "variance": "dotted"}
    y_label_dict = {"se": "左図：平均二乗誤差", "bias": "中図：二乗バイアス", "variance": "右図：バリアンス"}

    # import open bandit pipeline (obp)
    import obp
    from obp.dataset import (
        SyntheticBanditDatasetWithActionEmbeds as SyntheticBanditDataset,
    )
    from obp.dataset import (
        logistic_polynomial_reward_function,
    )
    from obp.ope import InverseProbabilityWeighting as IPS
    from obp.ope import OffPolicyEvaluation
    from utils import aggregate_simulation_results, aggregate_simulation_results_lam, eps_greedy_policy

    return (
        DataFrame,
        IPS,
        OffPolicyEvaluation,
        SyntheticBanditDataset,
        aggregate_simulation_results,
        aggregate_simulation_results_lam,
        check_random_state,
        eps_greedy_policy,
        japanize_matplotlib,
        linestyle_dict,
        logistic_polynomial_reward_function,
        np,
        obp,
        pd,
        plt,
        sns,
        tqdm,
        warnings,
        y_label_dict,
    )


@app.cell
def __(obp):
    print(obp.__version__)
    return


@app.cell
def __(mo):
    mo.md(
        r"""### ハイパーパラメータの値$\lambda$を変化させたときのCIPS推定量の平均二乗誤差・二乗バイアス・バリアンスの挙動"""
    )
    return


@app.cell
def __(check_random_state):
    ## シミュレーション設定
    num_runs = 50  # シミュレーションの繰り返し回数
    dim_context = 10  # 特徴量xの次元
    n_actions = 100  # 行動数, |A|
    beta = -5  # データ収集方策のパラメータ
    random_state = 12345
    test_data_size = 100000  # 評価方策の真の性能を近似するためのテストデータのサイズ
    random_ = check_random_state(random_state)
    num_data_list = [250, 500, 1000, 2000, 4000, 8000]
    return (
        beta,
        dim_context,
        n_actions,
        num_data_list,
        num_runs,
        random_,
        random_state,
        test_data_size,
    )


@app.cell
def __(
    IPS,
    OffPolicyEvaluation,
    SyntheticBanditDataset,
    aggregate_simulation_results_lam,
    beta,
    dim_context,
    eps_greedy_policy,
    logistic_polynomial_reward_function,
    n_actions,
    num_runs,
    pd,
    random_,
    random_state,
    test_data_size,
    tqdm,
):
    result_df_list_lambda = []
    for num_data in [250, 500, 2000]:
        ## 人工データ生成クラス
        dataset_lambda = SyntheticBanditDataset(
            n_actions=n_actions,
            dim_context=dim_context,
            action_context=random_.normal(size=(n_actions, 10)),
            beta=beta,
            reward_type="continuous",
            reward_function=logistic_polynomial_reward_function,
            random_state=random_state,
        )

        ## 評価方策の真の性能(policy value)を近似するためのテストデータ
        test_data_lambda = dataset_lambda.obtain_batch_bandit_feedback(n_rounds=test_data_size)

        ## 評価方策の真の性能(policy value)を近似
        policy_value_lambda = dataset_lambda.calc_ground_truth_policy_value(
            expected_reward=test_data_lambda["expected_reward"],
            action_dist=eps_greedy_policy(test_data_lambda["expected_reward"]),
        )

        estimated_policy_value_list_lambda = []
        for _ in tqdm(range(num_runs), desc=f"num_data={num_data}..."):
            ## データ収集方策が形成する分布に従いログデータを生成
            offline_logged_data_lambda = dataset_lambda.obtain_batch_bandit_feedback(n_rounds=num_data)

            ## ログデータ上における評価方策の行動選択確率を計算
            pi_lambda = eps_greedy_policy(offline_logged_data_lambda["expected_reward"])

            ## ログデータを用いてオフ方策評価を実行する
            ope_lambda = OffPolicyEvaluation(
                bandit_feedback=offline_logged_data_lambda,
                ope_estimators=[IPS(lambda_=lam, estimator_name=lam) for lam in range(0, 500)],
            )
            estimated_policy_values_lambda = ope_lambda.estimate_policy_values(action_dist=pi_lambda)
            estimated_policy_value_list_lambda.append(estimated_policy_values_lambda)

        ## シミュレーション結果を集計する
        result_df_list_lambda.append(
            aggregate_simulation_results_lam(
                estimated_policy_value_list_lambda,
                policy_value_lambda,
                "num_data",
                num_data,
            )
        )
    result_df_lambda = pd.concat(result_df_list_lambda).reset_index(level=0)
    result_df_lambda = result_df_lambda.groupby(["lam", "num_data"]).mean().reset_index()
    return (
        dataset_lambda,
        estimated_policy_value_list_lambda,
        estimated_policy_values_lambda,
        offline_logged_data_lambda,
        ope_lambda,
        pi_lambda,
        policy_value_lambda,
        result_df_lambda,
        result_df_list_lambda,
        test_data_lambda,
    )


@app.cell
def __(mo):
    mo.md(r"""## 図1.21""")
    return


@app.cell
def __(linestyle_dict, plt, result_df_lambda, sns, y_label_dict):
    fig1_21, ax_list_21 = plt.subplots(1, 3, figsize=(35, 10), tight_layout=True)
    for i21, num_data in enumerate([250, 500, 2000]):
        ax21 = ax_list_21[i21]
        result_df_21 = result_df_lambda.query(f"num_data == {num_data}").reset_index()
        optimal_hyperparam_index = result_df_21.se.argmin()
        optimal_hyperparam = result_df_21.loc[optimal_hyperparam_index, "lam"]
        for y21 in ["se", "bias", "variance"]:
            sns.lineplot(
                linewidth=9,
                legend=False,
                linestyle=linestyle_dict[y21],
                x="lam",
                y=y21,
                ax=ax21,
                data=result_df_21,
            )
            ax21.set_title(f"ログデータのサイズ$n=${num_data}", fontsize=50)
            # yaxis
            ax21.set_ylabel("")
            ax21.set_ylim(0.0, 0.65)
            ax21.tick_params(axis="y", labelsize=35)
            ax21.set_yticks([0.0, 0.3, 0.6])
            ax21.yaxis.set_label_coords(-0.1, 0.5)
            # xaxis
            if i21 == 1:
                ax21.set_xlabel(r"ハイパーパラメータ$\lambda$の値", fontsize=50)
            else:
                ax21.set_xlabel(r"", fontsize=40)
            ax21.set_xticks([0, 100, 200, 300, 400, 500])
            ax21.set_xticklabels([0, 100, 200, 300, 400, 500], fontsize=30)
            ax21.xaxis.set_label_coords(0.5, -0.12)
            ax21.vlines(optimal_hyperparam, 0, 0.65, color="black", linewidth=5, linestyles="dotted")
            if num_data > 500:
                ax21.text(optimal_hyperparam - 260, 0.55, f"最適な$\lambda=${optimal_hyperparam}", fontsize=50)
            else:
                ax21.text(optimal_hyperparam + 20, 0.55, f"最適な$\lambda=${optimal_hyperparam}", fontsize=50)
    fig1_21.legend(
        ["平均二乗誤差 (MSE)", "二乗バイアス", "バリアンス"],
        fontsize=50,
        bbox_to_anchor=(0.5, 1.12),
        ncol=4,
        loc="center",
    )
    fig1_21
    return (
        ax21,
        ax_list_21,
        fig1_21,
        i21,
        optimal_hyperparam,
        optimal_hyperparam_index,
        result_df_21,
        y21,
    )


@app.cell
def __(mo):
    mo.md(
        r"""### (データ収集方策が収集した)ログデータのサイズ$n$を変化させたときのCIPS推定量の平均二乗誤差・二乗バイアス・バリアンスの挙動"""
    )
    return


@app.cell
def __(
    IPS,
    OffPolicyEvaluation,
    SyntheticBanditDataset,
    aggregate_simulation_results,
    beta,
    dim_context,
    eps_greedy_policy,
    logistic_polynomial_reward_function,
    n_actions,
    num_data_list,
    num_runs,
    pd,
    random_,
    random_state,
    test_data_size,
    tqdm,
):
    result_df_list_size = []
    for num_data in num_data_list:
        ## 人工データ生成クラス
        dataset_size = SyntheticBanditDataset(
            n_actions=n_actions,
            dim_context=dim_context,
            action_context=random_.normal(size=(n_actions, 10)),
            beta=beta,
            reward_type="continuous",
            reward_function=logistic_polynomial_reward_function,
            random_state=random_state,
        )

        ## 評価方策の真の性能(policy value)を近似するためのテストデータ
        test_data_size_exp = dataset_size.obtain_batch_bandit_feedback(n_rounds=test_data_size)

        ## 評価方策の真の性能(policy value)を近似
        policy_value_size = dataset_size.calc_ground_truth_policy_value(
            expected_reward=test_data_size_exp["expected_reward"],
            action_dist=eps_greedy_policy(test_data_size_exp["expected_reward"]),
        )

        estimated_policy_value_list_size = []
        for _ in tqdm(range(num_runs), desc=f"num_data={num_data}..."):
            ## データ収集方策が形成する分布に従いログデータを生成
            offline_logged_data_size = dataset_size.obtain_batch_bandit_feedback(n_rounds=num_data)

            ## ログデータ上における評価方策の行動選択確率を計算
            pi_size = eps_greedy_policy(offline_logged_data_size["expected_reward"])

            ## ログデータを用いてオフ方策評価を実行する
            ope_size = OffPolicyEvaluation(
                bandit_feedback=offline_logged_data_size,
                ope_estimators=[
                    IPS(lambda_=50, estimator_name="50"),
                    IPS(lambda_=150, estimator_name="150"),
                    IPS(estimator_name="infty"),
                ],
            )
            estimated_policy_values_size = ope_size.estimate_policy_values(action_dist=pi_size)
            estimated_policy_value_list_size.append(estimated_policy_values_size)

        ## シミュレーション結果を集計する
        result_df_list_size.append(
            aggregate_simulation_results(
                estimated_policy_value_list_size,
                policy_value_size,
                "num_data",
                num_data,
            )
        )
    result_df_size = pd.concat(result_df_list_size).reset_index(level=0)
    return (
        dataset_size,
        estimated_policy_value_list_size,
        estimated_policy_values_size,
        offline_logged_data_size,
        ope_size,
        pi_size,
        policy_value_size,
        result_df_list_size,
        result_df_size,
        test_data_size_exp,
    )


@app.cell
def __(mo):
    mo.md(r"""## 図1.22""")
    return


@app.cell
def __(num_data_list, plt, result_df_size, sns, y_label_dict):
    fig1_22, ax_list_22 = plt.subplots(1, 3, figsize=(35, 10), tight_layout=True)
    for i22, y22 in enumerate(["se", "bias", "variance"]):
        ax22 = ax_list_22[i22]
        sns.lineplot(
            markers=True,
            markersize=40,
            linewidth=5,
            legend=False,
            style="est",
            x="num_data",
            y=y22,
            hue="est",
            ax=ax22,
            ci=None,
            data=result_df_size,
        )
        ax22.set_title(y_label_dict[y22], fontsize=50)
        # yaxis
        ax22.set_ylabel("")
        ax22.set_ylim(0.0, 0.725)
        ax22.tick_params(axis="y", labelsize=30)
        ax22.set_yticks([0.0, 0.35, 0.7])
        ax22.yaxis.set_label_coords(-0.1, 0.5)
        # xaxis
        ax22.set_xscale("log")
        if i22 == 1:
            ax22.set_xlabel(r"ログデータのサイズ$n$", fontsize=50)
        else:
            ax22.set_xlabel(r"", fontsize=40)
        ax22.set_xticks(num_data_list)
        ax22.set_xticklabels(num_data_list, fontsize=30)
        ax22.xaxis.set_label_coords(0.5, -0.12)
    fig1_22.legend(
        [r"$\lambda=50$", r"$\lambda=150$", r"$\lambda=\infty$ (= IPS)"],
        fontsize=50,
        bbox_to_anchor=(0.5, 1.15),
        ncol=4,
        loc="center",
    )
    fig1_22
    return ax22, ax_list_22, fig1_22, i22, y22


@app.cell
def __():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
