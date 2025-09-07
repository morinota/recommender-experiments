import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # 1.2.2 ~ 1.2.4 Direct Method (DM), Inverse Propensity Score (IPS), Doubly Robust (DR)推定量
    参考文献
    - Miroslav Dudik, John Langford, and Lihong Li. [Doubly Robust Policy Evaluation and Learning](https://arxiv.org/abs/1103.4601). ICML2011.
    """
    )
    return


@app.cell
def _():
    import warnings

    warnings.filterwarnings("ignore")

    import japanize_matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from pandas import DataFrame
    from sklearn.linear_model import LogisticRegression, Ridge
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
        IPS,
        MLPRegressor,
        OffPolicyEvaluation,
        Ridge,
        RegressionModel,
        SyntheticBanditDataset,
        aggregate_simulation_results,
        check_random_state,
        eps_greedy_policy,
        logistic_polynomial_reward_function,
        obp,
        pd,
        plt,
        sns,
        tqdm,
        y_label_dict,
    )


@app.cell
def _(obp):
    print(obp.__version__)
    return


@app.cell
def _(mo):
    mo.md(
        r"""### (データ収集方策が収集した)ログデータのサイズ$n$を変化させたときのDM・IPS・DR推定量の平均二乗誤差・二乗バイアス・バリアンスの挙動"""
    )
    return


@app.cell
def _(check_random_state):
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
def _(
    DM,
    DR,
    IPS,
    MLPRegressor,
    OffPolicyEvaluation,
    RegressionModel,
    Ridge,
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

            ## 期待報酬関数に対する推定モデル\hat{q}(x,a)を得る
            reg_model = RegressionModel(
                n_actions=dataset.n_actions,
                base_model=Ridge(alpha=1.0, random_state=random_state),
            )
            estimated_rewards_lr = reg_model.fit_predict(
                context=offline_logged_data["context"],  # context; x
                action=offline_logged_data["action"],  # action; a
                reward=offline_logged_data["reward"],  # reward; r
                random_state=random_state,
            )
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
                    DR(estimator_name="DR"),
                    DM(estimator_name="lr"),
                    DM(estimator_name="mlp"),
                ],
            )
            estimated_policy_values = ope.estimate_policy_values(
                action_dist=pi,  # \pi(a|x)
                estimated_rewards_by_reg_model={
                    "DR": estimated_rewards_mlp,
                    "lr": estimated_rewards_lr,
                    "mlp": estimated_rewards_mlp,
                },
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
    return (result_df,)


@app.cell
def _(mo):
    mo.md(r"""## 図1.9""")
    return


@app.cell
def _(num_data_list, plt, result_df, sns, y_label_dict):
    fig1_9, ax_list = plt.subplots(1, 3, figsize=(35, 10), tight_layout=True)
    for i, y in enumerate(["se", "bias", "variance"]):
        ax = ax_list[i]
        sns.lineplot(
            markers=True,
            markersize=40,
            linewidth=5,
            legend=False,
            style="est",
            x="num_data",
            y=y,
            hue="est",
            ax=ax,
            ci=None,
            palette=["tab:grey", "tab:purple"],
            data=result_df.query("est == 'lr' or est == 'mlp'"),
        )
        ax.set_title(y_label_dict[y], fontsize=50)
        # yaxis
        ax.set_ylabel("")
        ax.set_ylim(0.0, 0.04)
        ax.tick_params(axis="y", labelsize=30)
        ax.set_yticks([0.0, 0.02, 0.04])
        ax.yaxis.set_label_coords(-0.1, 0.5)
        # xaxis
        ax.set_xscale("log")
        if i == 1:
            ax.set_xlabel(r"ログデータのサイズ$n$", fontsize=50)
        else:
            ax.set_xlabel(r"", fontsize=40)
        ax.set_xticks(num_data_list)
        ax.set_xticklabels(num_data_list, fontsize=30)
        ax.xaxis.set_label_coords(0.5, -0.12)
    fig1_9.legend(
        ["DM (リッジ回帰)", "DM (ニューラルネットワーク)"],
        fontsize=50,
        bbox_to_anchor=(0.5, 1.15),
        ncol=4,
        loc="center",
    )
    fig1_9
    return


@app.cell
def _(mo):
    mo.md(r"""## 図1.13""")
    return


@app.cell
def _(num_data_list, plt, result_df, sns, y_label_dict):
    fig1_13, ax_list_13 = plt.subplots(1, 3, figsize=(35, 10), tight_layout=True)
    for i13, y13 in enumerate(["se", "bias", "variance"]):
        ax13 = ax_list_13[i13]
        sns.lineplot(
            markers=True,
            markersize=40,
            linewidth=5,
            legend=False,
            style="est",
            x="num_data",
            y=y13,
            hue="est",
            ax=ax13,
            ci=None,
            palette=["tab:red", "tab:purple"],
            data=result_df.query("est == 'IPS' or est == 'mlp'"),
        )
        ax13.set_title(y_label_dict[y13], fontsize=50)
        # yaxis
        ax13.set_ylabel("")
        ax13.set_ylim(0.0, 0.085)
        ax13.tick_params(axis="y", labelsize=30)
        ax13.set_yticks([0.0, 0.04, 0.08])
        ax13.yaxis.set_label_coords(-0.1, 0.5)
        # xaxis
        ax13.set_xscale("log")
        if i13 == 1:
            ax13.set_xlabel(r"ログデータのサイズ$n$", fontsize=50)
        else:
            ax13.set_xlabel(r"", fontsize=40)
        ax13.set_xticks(num_data_list)
        ax13.set_xticklabels(num_data_list, fontsize=30)
        ax13.xaxis.set_label_coords(0.5, -0.12)
    fig1_13.legend(
        ["IPS", "DM (ニューラルネットワーク)"],
        fontsize=50,
        bbox_to_anchor=(0.5, 1.15),
        ncol=4,
        loc="center",
    )
    fig1_13
    return


@app.cell
def _(mo):
    mo.md(r"""## 図1.16""")
    return


@app.cell
def _(num_data_list, plt, result_df, sns, y_label_dict):
    fig1_16, ax_list_16 = plt.subplots(1, 3, figsize=(35, 10), tight_layout=True)
    for i16, y16 in enumerate(["se", "bias", "variance"]):
        ax16 = ax_list_16[i16]
        sns.lineplot(
            markers=True,
            markersize=40,
            linewidth=5,
            legend=False,
            style="est",
            x="num_data",
            y=y16,
            hue="est",
            ax=ax16,
            ci=None,
            palette=["tab:red", "tab:blue", "tab:purple"],
            data=result_df.query("est != 'lr'"),
        )
        ax16.set_title(y_label_dict[y16], fontsize=50)
        # yaxis
        ax16.set_ylabel("")
        ax16.set_ylim(0.0, 0.085)
        ax16.tick_params(axis="y", labelsize=30)
        ax16.set_yticks([0.0, 0.04, 0.08])
        ax16.yaxis.set_label_coords(-0.1, 0.5)
        # xaxis
        ax16.set_xscale("log")
        if i16 == 1:
            ax16.set_xlabel(r"ログデータのサイズ$n$", fontsize=50)
        else:
            ax16.set_xlabel(r"", fontsize=40)
        ax16.set_xticks(num_data_list)
        ax16.set_xticklabels(num_data_list, fontsize=30)
        ax16.xaxis.set_label_coords(0.5, -0.12)
    fig1_16.legend(
        ["IPS", "DR", "DM (ニューラルネットワーク)"],
        fontsize=50,
        bbox_to_anchor=(0.5, 1.15),
        ncol=4,
        loc="center",
    )
    fig1_16
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
