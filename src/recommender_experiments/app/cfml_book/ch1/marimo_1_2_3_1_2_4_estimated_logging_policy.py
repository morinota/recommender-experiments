import marimo

__generated_with = "0.9.24"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    mo.md(r"""# 1.2.3-1.2.4 データ収集方策を推定する場合""")
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
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.utils import check_random_state
    from tqdm import tqdm

    plt.style.use("ggplot")
    y_label_dict = {"se": "左図：平均二乗誤差", "bias": "中図：二乗バイアス", "variance": "右図：バリアンス"}

    # import open bandit pipeline (obp)
    import obp
    from obp.dataset import (
        SyntheticBanditDatasetWithActionEmbeds as SyntheticBanditDataset,
    )
    from obp.dataset import (
        logistic_polynomial_reward_function,
    )
    from obp.ope import (
        DirectMethod as DM,
    )
    from obp.ope import (
        DoublyRobust as DR,
    )
    from obp.ope import (
        InverseProbabilityWeighting as IPS,
    )
    from obp.ope import (
        OffPolicyEvaluation,
        RegressionModel,
    )
    from utils import aggregate_simulation_results, eps_greedy_policy

    return (
        DM,
        DR,
        DataFrame,
        IPS,
        LogisticRegression,
        MLPClassifier,
        MLPRegressor,
        OffPolicyEvaluation,
        RegressionModel,
        SyntheticBanditDataset,
        aggregate_simulation_results,
        check_random_state,
        eps_greedy_policy,
        japanize_matplotlib,
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
        r"""### ログデータのサイズ$n$を変化させたときの真のデータ収集方策を用いる場合とデータ収集方策を推定する場合のIPS・DR推定量の平均二乗誤差・二乗バイアス・バリアンスの挙動"""
    )
    return


@app.cell
def __(check_random_state):
    ## シミュレーション設定
    num_runs = 500  # シミュレーションの繰り返し回数
    dim_context = 10  # 特徴量xの次元
    n_actions = 20  # 行動数, |A|
    beta = -3  # データ収集方策のパラメータ
    test_data_size = 100000  # 評価方策の真の性能を近似するためのテストデータのサイズ
    random_state = 12345
    random_ = check_random_state(random_state)
    num_data_list = [250, 500, 1000, 2000, 4000, 8000]  # データ収集方策が収集したログデータのサイズ
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
    DR,
    IPS,
    LogisticRegression,
    MLPRegressor,
    OffPolicyEvaluation,
    RegressionModel,
    SyntheticBanditDataset,
    aggregate_simulation_results,
    beta,
    dim_context,
    eps_greedy_policy,
    logistic_polynomial_reward_function,
    n_actions,
    np,
    num_data_list,
    num_runs,
    pd,
    random_,
    random_state,
    test_data_size,
    tqdm,
):
    result_df_list = []
    for num_data in num_data_list:
        ## 人工データ生成クラス
        dataset = SyntheticBanditDataset(
            n_actions=n_actions,
            dim_context=dim_context,
            action_context=random_.normal(size=(n_actions, 10)),
            beta=beta,
            reward_type="continuous",
            reward_function=logistic_polynomial_reward_function,
            random_state=random_state,
        )

        ## 評価方策の真の性能(policy value)を近似するためのテストデータ
        test_data = dataset.obtain_batch_bandit_feedback(n_rounds=test_data_size)

        ## 評価方策の真の性能(policy value)を近似
        policy_value = dataset.calc_ground_truth_policy_value(
            expected_reward=test_data["expected_reward"],
            action_dist=eps_greedy_policy(test_data["expected_reward"]),
        )

        estimated_policy_value_list = []
        for _ in tqdm(range(num_runs), desc=f"num_data={num_data}..."):
            ## データ収集方策が形成する分布に従いログデータを生成
            offline_logged_data = dataset.obtain_batch_bandit_feedback(n_rounds=num_data)

            ## ログデータ上における評価方策の行動選択確率を計算
            pi = eps_greedy_policy(offline_logged_data["expected_reward"])

            ## ログデータを用いてデータ収集方策を推定
            lr = LogisticRegression(C=100, random_state=random_state)
            lr.fit(offline_logged_data["context"], offline_logged_data["action"])
            estimated_pi0 = lr.predict_proba(offline_logged_data["context"])[
                np.arange(num_data), offline_logged_data["action"]
            ]

            ## 期待報酬関数に対する推定モデル\hat{q}(x,a)を得る
            reg_model = RegressionModel(
                n_actions=dataset.n_actions,
                base_model=MLPRegressor(hidden_layer_sizes=(10, 10), random_state=random_state, max_iter=1000),
            )
            estimated_rewards_mlp = reg_model.fit_predict(
                context=offline_logged_data["context"],  # context; x
                action=offline_logged_data["action"],  # action; a
                reward=offline_logged_data["reward"],  # reward; r
                random_state=random_state,
            )

            ## ログデータを用いてオフ方策評価を実行する
            ope = OffPolicyEvaluation(
                bandit_feedback=offline_logged_data,
                ope_estimators=[
                    IPS(estimator_name="IPS"),
                    IPS(estimator_name="IPS (estimated)", use_estimated_pscore=True),
                    DR(estimator_name="DR"),
                    DR(estimator_name="DR (estimated)", use_estimated_pscore=True),
                ],
            )
            estimated_policy_values = ope.estimate_policy_values(
                action_dist=pi,  # \pi(a|x)
                estimated_rewards_by_reg_model=estimated_rewards_mlp,
                estimated_pscore=estimated_pi0,
            )
            estimated_policy_value_list.append(estimated_policy_values)

        ## シミュレーション結果を集計する
        result_df_list.append(
            aggregate_simulation_results(
                estimated_policy_value_list,
                policy_value,
                "num_data",
                num_data,
            )
        )
    result_df = pd.concat(result_df_list).reset_index(level=0)
    return (
        dataset,
        estimated_policy_value_list,
        estimated_policy_values,
        estimated_pi0,
        estimated_rewards_mlp,
        lr,
        offline_logged_data,
        ope,
        pi,
        policy_value,
        reg_model,
        result_df,
        result_df_list,
        test_data,
    )


@app.cell
def __(mo):
    mo.md(r"""## 図1.15""")
    return


@app.cell
def __(num_data_list, plt, result_df, sns, y_label_dict):
    fig1_15, ax_list_15 = plt.subplots(1, 3, figsize=(35, 10), tight_layout=True)
    for i15, y15 in enumerate(["se", "bias", "variance"]):
        ax15 = ax_list_15[i15]
        sns.lineplot(
            markers=True,
            markersize=40,
            linewidth=5,
            legend=False,
            style="est",
            x="num_data",
            y=y15,
            hue="est",
            ax=ax15,
            ci=None,
            palette=["tab:red", "tab:grey"],
            data=result_df.query("est == 'IPS' or est == 'IPS (estimated)'"),
        )
        ax15.set_title(y_label_dict[y15], fontsize=50)
        # yaxis
        ax15.set_ylabel("")
        ax15.set_ylim(0.0, 0.14)
        ax15.tick_params(axis="y", labelsize=30)
        ax15.set_yticks([0.0, 0.06, 0.12])
        ax15.yaxis.set_label_coords(-0.1, 0.5)
        # xaxis
        ax15.set_xscale("log")
        if i15 == 1:
            ax15.set_xlabel(r"ログデータのサイズ$n$", fontsize=50)
        else:
            ax15.set_xlabel(r"", fontsize=40)
        ax15.set_xticks(num_data_list)
        ax15.set_xticklabels(num_data_list, fontsize=30)
        ax15.xaxis.set_label_coords(0.5, -0.12)
    fig1_15.legend(
        ["IPS (真のデータ収集方策を用いた場合)", "IPS (データ収集方策を推定した場合)"],
        fontsize=50,
        bbox_to_anchor=(0.5, 1.15),
        ncol=4,
        loc="center",
    )
    fig1_15
    return ax_list_15, fig1_15, i15, y15


@app.cell
def __(mo):
    mo.md(r"""## 図1.19""")
    return


@app.cell
def __(num_data_list, plt, result_df, sns, y_label_dict):
    fig1_19, ax_list_19 = plt.subplots(1, 3, figsize=(35, 10), tight_layout=True)
    for i19, y19 in enumerate(["se", "bias", "variance"]):
        ax19 = ax_list_19[i19]
        sns.lineplot(
            markers=True,
            markersize=40,
            linewidth=5,
            legend=False,
            style="est",
            x="num_data",
            y=y19,
            hue="est",
            ax=ax19,
            ci=None,
            palette=["tab:blue", "tab:purple"],
            data=result_df.query("est == 'DR' or est == 'DR (estimated)'"),
        )
        ax19.set_title(y_label_dict[y19], fontsize=50)
        # yaxis
        ax19.set_ylabel("")
        ax19.set_ylim(0.0, 0.046)
        ax19.tick_params(axis="y", labelsize=30)
        ax19.set_yticks([0.0, 0.015, 0.03, 0.045])
        ax19.yaxis.set_label_coords(-0.1, 0.5)
        # xaxis
        ax19.set_xscale("log")
        if i19 == 1:
            ax19.set_xlabel(r"ログデータのサイズ$n$", fontsize=50)
        else:
            ax19.set_xlabel(r"", fontsize=40)
        ax19.set_xticks(num_data_list)
        ax19.set_xticklabels(num_data_list, fontsize=30)
        ax19.xaxis.set_label_coords(0.5, -0.12)
    fig1_19.legend(
        ["DR (真のデータ収集方策を用いた場合)", "DR (データ収集方策を推定した場合)"],
        fontsize=50,
        bbox_to_anchor=(0.5, 1.15),
        ncol=4,
        loc="center",
    )
    fig1_19
    return ax_list_19, fig1_19, i19, y19


@app.cell
def __(mo):
    mo.md(r"""## 図1.20""")
    return


@app.cell
def __(num_data_list, plt, result_df, sns, y_label_dict):
    fig1_20, ax_list_20 = plt.subplots(1, 3, figsize=(35, 10), tight_layout=True)
    for i20, y20 in enumerate(["se", "bias", "variance"]):
        ax20 = ax_list_20[i20]
        sns.lineplot(
            markers=True,
            markersize=40,
            linewidth=5,
            legend=False,
            style="est",
            x="num_data",
            y=y20,
            hue="est",
            ax=ax20,
            ci=None,
            palette=["tab:grey", "tab:purple"],
            data=result_df.query("est == 'IPS (estimated)' or est == 'DR (estimated)'"),
        )
        ax20.set_title(y_label_dict[y20], fontsize=50)
        # yaxis
        ax20.set_ylabel("")
        ax20.set_ylim(0.0, 0.14)
        ax20.tick_params(axis="y", labelsize=30)
        ax20.set_yticks([0.0, 0.06, 0.12])
        ax20.yaxis.set_label_coords(-0.1, 0.5)
        # xaxis
        ax20.set_xscale("log")
        if i20 == 1:
            ax20.set_xlabel(r"ログデータのサイズ$n$", fontsize=50)
        else:
            ax20.set_xlabel(r"", fontsize=40)
        ax20.set_xticks(num_data_list)
        ax20.set_xticklabels(num_data_list, fontsize=30)
        ax20.xaxis.set_label_coords(0.5, -0.12)
    fig1_20.legend(
        ["IPS (データ収集方策を推定した場合)", "DR (データ収集方策を推定した場合)"],
        fontsize=50,
        bbox_to_anchor=(0.5, 1.15),
        ncol=4,
        loc="center",
    )
    fig1_20
    return ax_list_20, fig1_20, i20, y20


@app.cell
def __():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
