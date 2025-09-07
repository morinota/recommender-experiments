import marimo

__generated_with = "0.9.24"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    mo.md(r"""# 1.3.3 Switch Doubly Robust(Switch-DR)推定量
    参考文献
    - Yu-Xiang Wang, Alekh Agarwal, and Miroslav Dudik. [Optimal and Adaptive Off-policy Evaluation in Contextual Bandits](https://arxiv.org/abs/1612.01205). ICML2017.""")
    return


@app.cell
def __():
    import warnings
    warnings.filterwarnings('ignore')

    import japanize_matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from pandas import DataFrame
    from sklearn.linear_model import Ridge
    from sklearn.utils import check_random_state
    from tqdm import tqdm
    plt.style.use('ggplot')
    linestyle_dict = {"se": "-", "bias": "--", "variance": "dotted"}
    y_label_dict = {"se": "左図：平均二乗誤差", "bias": "中図：二乗バイアス", "variance": "右図：バリアンス"}

    # import open bandit pipeline (obp)
    import obp
    from obp.dataset import (
        SyntheticBanditDatasetWithActionEmbeds as SyntheticBanditDataset,
    )
    from obp.dataset import (
        logistic_polynomial_reward_function,
        polynomial_reward_function,
    )
    from obp.ope import OffPolicyEvaluation, RegressionModel
    from obp.ope import SwitchDoublyRobust as SwitchDR
    from utils import aggregate_simulation_results, aggregate_simulation_results_lam, eps_greedy_policy
    return (
        DataFrame,
        OffPolicyEvaluation,
        RegressionModel,
        Ridge,
        SwitchDR,
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
        polynomial_reward_function,
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
    mo.md(r"""### ハイパーパラメータの値$\lambda$を変化させたときのSwitch-DR推定量の平均二乗誤差・二乗バイアス・バリアンスの挙動""")
    return


@app.cell
def __(check_random_state):
    ## シミュレーション設定
    num_runs = 1000  # シミュレーションの繰り返し回数
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
    OffPolicyEvaluation,
    RegressionModel,
    Ridge,
    SwitchDR,
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
    result_df_list_24 = []
    for num_data in [500, 1000, 4000]:
        ## 人工データ生成クラス
        dataset_24 = SyntheticBanditDataset(
            n_actions=n_actions,
            dim_context=10,
            action_context=random_.normal(size=(n_actions, 10)),
            beta=-5,
            reward_type="continuous",
            reward_function=logistic_polynomial_reward_function,
            random_state=random_state,
        )

        ## 評価方策の真の性能(policy value)を近似するためのテストデータ
        test_data_24 = dataset_24.obtain_batch_bandit_feedback(n_rounds=test_data_size)

        ## 評価方策の真の性能(policy value)を近似
        policy_value_24 = dataset_24.calc_ground_truth_policy_value(
            expected_reward=test_data_24["expected_reward"],
            action_dist=eps_greedy_policy(test_data_24["expected_reward"]),
        )

        estimated_policy_value_list_24 = []
        for _ in tqdm(range(num_runs), desc=f"num_data={num_data}..."):
            ## データ収集方策が形成する分布に従いログデータを生成
            offline_logged_data_24 = dataset_24.obtain_batch_bandit_feedback(
                n_rounds=num_data
            )

            ## ログデータ上における評価方策の行動選択確率を計算
            pi_24 = eps_greedy_policy(offline_logged_data_24["expected_reward"])

            ## 期待報酬関数に対する推定モデル\hat{q}(x,a)を得る
            reg_model_24 = RegressionModel(
                n_actions=dataset_24.n_actions,
                base_model=Ridge(alpha=1.0, random_state=random_state),
            )
            estimated_rewards_mlp_24 = reg_model_24.fit_predict(
                context=offline_logged_data_24["context"], # context; x
                action=offline_logged_data_24["action"], # action; a
                reward=offline_logged_data_24["reward"], # reward; r
                random_state=random_state,
            )

            ## ログデータを用いてオフ方策評価を実行する
            ope_24 = OffPolicyEvaluation(
                bandit_feedback=offline_logged_data_24,
                ope_estimators=[SwitchDR(lambda_=lam, estimator_name=lam) for lam in range(0, 1000, 2)]
            )
            estimated_policy_values_24 = ope_24.estimate_policy_values(
                action_dist=pi_24,
                estimated_rewards_by_reg_model=estimated_rewards_mlp_24,
            )
            estimated_policy_value_list_24.append(estimated_policy_values_24)

        ## シミュレーション結果を集計する
        result_df_list_24.append(
            aggregate_simulation_results_lam(
                estimated_policy_value_list_24, policy_value_24, "num_data", num_data,
            )
        )
    result_df_24 = pd.concat(result_df_list_24).reset_index(level=0)
    result_df_24 = result_df_24.groupby(["lam", "num_data"]).mean().reset_index()
    return (
        dataset_24,
        estimated_policy_value_list_24,
        estimated_policy_values_24,
        estimated_rewards_mlp_24,
        offline_logged_data_24,
        ope_24,
        pi_24,
        policy_value_24,
        reg_model_24,
        result_df_24,
        result_df_list_24,
        test_data_24,
    )


@app.cell
def __(mo):
    mo.md(r"""## 図1.24""")
    return


@app.cell
def __(linestyle_dict, plt, result_df_24, sns, y_label_dict):
    fig1_24, ax_list_24 = plt.subplots(1, 3, figsize=(35, 10), tight_layout=True)
    for i24, num_data in enumerate([500, 1000, 4000]):
        ax24 = ax_list_24[i24]
        result_df_24_filtered = result_df_24.query(f"num_data == {num_data}").reset_index()
        optimal_hyperparam_index_24 = result_df_24_filtered.se.argmin()
        optimal_hyperparam_24 = result_df_24_filtered.loc[optimal_hyperparam_index_24, "lam"]
        for y24 in ["se", "bias", "variance"]:
            sns.lineplot(
                linewidth=9,
                legend=False,
                linestyle=linestyle_dict[y24],
                x="lam",
                y=y24,
                ax=ax24,
                data=result_df_24_filtered,
            )
            ax24.set_title(f"ログデータのサイズ$n=${num_data}", fontsize=50)
            # yaxis
            ax24.set_ylabel("")
            ax24.set_ylim(0.0, 0.15)
            ax24.tick_params(axis="y", labelsize=35)
            ax24.set_yticks([0.0, 0.075, 0.15])
            ax24.yaxis.set_label_coords(-0.1, 0.5)
            # xaxis
            if i24 == 1:
                ax24.set_xlabel(r"ハイパーパラメータ$\lambda$の値", fontsize=50)
            else:
                ax24.set_xlabel(r"", fontsize=40)
            ax24.set_xticks([0, 200, 400, 600, 800, 1000])
            ax24.set_xticklabels([0, 200, 400, 600, 800, 1000], fontsize=30)
            ax24.xaxis.set_label_coords(0.5, -0.12)
            ax24.vlines(optimal_hyperparam_24, 0, 0.7, color='black', linewidth=5, linestyles='dotted')
            ax24.text(optimal_hyperparam_24+20, 0.1, f"最適な$\lambda=${optimal_hyperparam_24}", fontsize=50)
    fig1_24.legend(
        ["平均二乗誤差 (MSE)", "二乗バイアス", "バリアンス"],
        fontsize=50, bbox_to_anchor=(0.5, 1.12), ncol=4, loc="center",
    )
    fig1_24
    return (
        ax24,
        ax_list_24,
        fig1_24,
        i24,
        optimal_hyperparam_24,
        optimal_hyperparam_index_24,
        result_df_24_filtered,
        y24,
    )


@app.cell
def __(mo):
    mo.md(r"""### (データ収集方策が収集した)ログデータのサイズ$n$を変化させたときのSwitch-DR推定量の平均二乗誤差・二乗バイアス・バリアンスの挙動""")
    return


@app.cell
def __(
    OffPolicyEvaluation,
    RegressionModel,
    Ridge,
    SwitchDR,
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
    result_df_list_25 = []
    for num_data in num_data_list:
        ## 人工データ生成クラス
        dataset_25 = SyntheticBanditDataset(
            n_actions=n_actions,
            dim_context=dim_context,
            action_context=random_.normal(size=(n_actions, 10)),
            beta=beta,
            reward_type="continuous",
            reward_function=logistic_polynomial_reward_function,
            random_state=random_state,
        )

        ## 評価方策の真の性能(policy value)を近似するためのテストデータ
        test_data_25 = dataset_25.obtain_batch_bandit_feedback(n_rounds=test_data_size)

        ## 評価方策の真の性能(policy value)を近似
        policy_value_25 = dataset_25.calc_ground_truth_policy_value(
            expected_reward=test_data_25["expected_reward"],
            action_dist=eps_greedy_policy(test_data_25["expected_reward"]),
        )

        estimated_policy_value_list_25 = []
        for _ in tqdm(range(num_runs), desc=f"num_data={num_data}..."):
            ## データ収集方策が形成する分布に従いログデータを生成
            offline_logged_data_25 = dataset_25.obtain_batch_bandit_feedback(
                n_rounds=num_data
            )

            ## ログデータ上における評価方策の行動選択確率を計算
            pi_25 = eps_greedy_policy(offline_logged_data_25["expected_reward"])

            ## 期待報酬関数に対する推定モデル\hat{q}(x,a)を得る
            reg_model_25 = RegressionModel(
                n_actions=dataset_25.n_actions,
                base_model=Ridge(alpha=1.0, random_state=random_state),
            )
            estimated_rewards_mlp_25 = reg_model_25.fit_predict(
                context=offline_logged_data_25["context"], # context; x
                action=offline_logged_data_25["action"], # action; a
                reward=offline_logged_data_25["reward"], # reward; r
                random_state=random_state,
            )

            ## ログデータを用いてオフ方策評価を実行する
            ope_25 = OffPolicyEvaluation(
                bandit_feedback=offline_logged_data_25,
                ope_estimators=[
                    SwitchDR(lambda_=0, estimator_name="0"),
                    SwitchDR(lambda_=num_data/4, estimator_name="adaptive"),
                    SwitchDR(estimator_name="infty"),
                ]
            )
            estimated_policy_values_25 = ope_25.estimate_policy_values(
                action_dist=pi_25,
                estimated_rewards_by_reg_model=estimated_rewards_mlp_25,
            )
            estimated_policy_value_list_25.append(estimated_policy_values_25)

        ## シミュレーション結果を集計する
        result_df_list_25.append(
            aggregate_simulation_results(
                estimated_policy_value_list_25, policy_value_25, "num_data", num_data,
            )
        )
    result_df_25 = pd.concat(result_df_list_25).reset_index(level=0)
    return (
        dataset_25,
        estimated_policy_value_list_25,
        estimated_policy_values_25,
        estimated_rewards_mlp_25,
        offline_logged_data_25,
        ope_25,
        pi_25,
        policy_value_25,
        reg_model_25,
        result_df_25,
        result_df_list_25,
        test_data_25,
    )


@app.cell
def __(mo):
    mo.md(r"""## 図1.25""")
    return


@app.cell
def __(num_data_list, plt, result_df_25, sns, y_label_dict):
    fig1_25, ax_list_25 = plt.subplots(1, 3, figsize=(35, 10), tight_layout=True)
    for i25, y25 in enumerate(["se", "bias", "variance"]):
        ax25 = ax_list_25[i25]
        sns.lineplot(
            markers=True,
            markersize=40,
            linewidth=5,
            legend=False,
            style="est",
            x="num_data",
            y=y25,
            hue="est",
            ax=ax25,
            ci=None,
            data=result_df_25,
        )
        ax25.set_title(y_label_dict[y25], fontsize=50)
        # yaxis
        ax25.set_ylabel("")
        ax25.set_ylim(0.0, 0.16)
        ax25.tick_params(axis="y", labelsize=30)
        ax25.set_yticks([0.0, 0.08, 0.16])
        ax25.yaxis.set_label_coords(-0.1, 0.5)
        # xaxis
        ax25.set_xscale("log")
        if i25 == 1:
            ax25.set_xlabel(r"ログデータのサイズ$n$", fontsize=50)
        else:
            ax25.set_xlabel(r"", fontsize=40)
        ax25.set_xticks(num_data_list)
        ax25.set_xticklabels(num_data_list, fontsize=30)
        ax25.xaxis.set_label_coords(0.5, -0.12)
    fig1_25.legend(
        [r"$\lambda=0$ (=DM)", r"$\lambda=n/4$（適応的な設定）", r"$\lambda=\infty$ (= DR)"],
        fontsize=50, bbox_to_anchor=(0.5, 1.15), ncol=4, loc="center",
    )
    fig1_25
    return ax25, ax_list_25, fig1_25, i25, y25


@app.cell
def __():
    import marimo as mo
    return mo,


if __name__ == "__main__":
    app.run()