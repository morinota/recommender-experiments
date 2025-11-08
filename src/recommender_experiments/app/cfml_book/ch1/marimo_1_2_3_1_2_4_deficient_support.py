import marimo

__generated_with = "0.9.24"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    mo.md(r"""# 1.2.3-1.2.4 共通サポートの仮定が満たされない場合
    参考文献
    - Noveen Sachdeva, Yi Su, and Thorsten Joachims. [Off-policy Bandits with Deficient Support](https://arxiv.org/abs/2006.09438). KDD2020.""")
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
    from sklearn.neural_network import MLPClassifier
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
    from recommender_experiments.app.cfml_book.common_utils import (
        aggregate_simulation_results,
        eps_greedy_policy,
    )

    return (
        DM,
        DR,
        DataFrame,
        IPS,
        MLPClassifier,
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
        r"""### 共通サポートが満たされない行動の割合$|U|/|A|$を変化させたときのIPS・DR推定量の平均二乗誤差・二乗バイアス・バリアンスの挙動"""
    )
    return


@app.cell
def __(check_random_state):
    ## シミュレーション設定
    num_runs = 50  # シミュレーションの繰り返し回数
    dim_context = 10  # 特徴量xの次元
    n_actions = 20  # 行動数, |A|
    beta = -3  # データ収集方策のパラメータ
    num_data = 2000  # ログデータのサイズ
    test_data_size = 100000  # 評価方策の真の性能を近似するためのテストデータのサイズ
    random_state = 12345
    random_ = check_random_state(random_state)
    n_def_actions_ratio_list = [0, 0.1, 0.2, 0.3, 0.4]
    return (
        beta,
        dim_context,
        n_actions,
        n_def_actions_ratio_list,
        num_data,
        num_runs,
        random_,
        random_state,
        test_data_size,
    )


@app.cell
def __(
    DM,
    DR,
    IPS,
    MLPClassifier,
    OffPolicyEvaluation,
    RegressionModel,
    SyntheticBanditDataset,
    aggregate_simulation_results,
    beta,
    dim_context,
    eps_greedy_policy,
    logistic_polynomial_reward_function,
    n_actions,
    n_def_actions_ratio_list,
    num_data,
    num_runs,
    pd,
    random_,
    random_state,
    test_data_size,
    tqdm,
):
    result_df_list = []
    for n_def_actions_ratio in n_def_actions_ratio_list:
        ## 人工データ生成クラス
        dataset = SyntheticBanditDataset(
            n_actions=n_actions,
            dim_context=dim_context,
            action_context=random_.normal(size=(n_actions, 10)),
            beta=beta,
            n_deficient_actions=int(n_def_actions_ratio * n_actions),
            reward_function=logistic_polynomial_reward_function,
            random_state=random_state,
        )

        ## 評価方策の真の性能(policy value)を近似するためのテストデータ
        test_data = dataset.obtain_batch_bandit_feedback(n_rounds=test_data_size)

        ## 評価方策の真の性能(policy value)を近似
        policy_value = dataset.calc_ground_truth_policy_value(
            expected_reward=test_data["expected_reward"],
            action_dist=eps_greedy_policy(
                test_data["expected_reward"], k=1, eps=0.1, return_normalized=True, rank_method=None, add_newaxis=True
            ),
        )

        estimated_policy_value_list = []
        for _ in tqdm(range(num_runs), desc=f"n_def_actions_ratio={n_def_actions_ratio}..."):
            ## データ収集方策が形成する分布に従いログデータを生成
            offline_logged_data = dataset.obtain_batch_bandit_feedback(n_rounds=num_data)

            ## ログデータ上における評価方策の行動選択確率を計算
            pi = eps_greedy_policy(
                offline_logged_data["expected_reward"],
                k=1,
                eps=0.1,
                return_normalized=True,
                rank_method=None,
                add_newaxis=True,
            )

            ## 期待報酬関数に対する推定モデル\hat{q}(x,a)を得る
            reg_model = RegressionModel(
                n_actions=dataset.n_actions,
                base_model=MLPClassifier(hidden_layer_sizes=(10, 10), random_state=random_state),
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
                    DM(estimator_name="DM"),
                ],
            )
            estimated_policy_values = ope.estimate_policy_values(
                action_dist=pi,  # \pi(a|x)
                estimated_rewards_by_reg_model=estimated_rewards_mlp,
            )
            estimated_policy_value_list.append(estimated_policy_values)

        ## シミュレーション結果を集計する
        result_df_list.append(
            aggregate_simulation_results(
                estimated_policy_value_list,
                policy_value,
                "n_def_actions_ratio",
                n_def_actions_ratio,
            )
        )
    result_df = pd.concat(result_df_list).reset_index(level=0)
    return (
        dataset,
        estimated_policy_value_list,
        estimated_policy_values,
        estimated_rewards_mlp,
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
    mo.md(r"""## 図1.14""")
    return


@app.cell
def __(n_def_actions_ratio_list, plt, result_df, sns, y_label_dict):
    fig1_14, ax_list_14 = plt.subplots(1, 3, figsize=(35, 10), tight_layout=True)
    for i, y in enumerate(["se", "bias", "variance"]):
        ax = ax_list_14[i]
        sns.lineplot(
            markers=True,
            markersize=40,
            linewidth=5,
            legend=False,
            style="est",
            x="n_def_actions_ratio",
            y=y,
            hue="est",
            ax=ax,
            palette=["tab:red", "tab:purple"],
            data=result_df.query("est != 'DR'"),
        )
        ax.set_title(y_label_dict[y], fontsize=50)
        # yaxis
        ax.set_ylabel("")
        ax.set_ylim(0.0, 0.065)
        ax.tick_params(axis="y", labelsize=30)
        ax.set_yticks([0.0, 0.03, 0.06])
        ax.yaxis.set_label_coords(-0.1, 0.5)
        # xaxis
        if i == 1:
            ax.set_xlabel(r"共通サポートの仮定が満たされない行動の割合$|U|/|A|$", fontsize=50)
        else:
            ax.set_xlabel("", fontsize=40)
        ax.set_xticks(n_def_actions_ratio_list)
        ax.set_xticklabels(n_def_actions_ratio_list, fontsize=35)
        ax.xaxis.set_label_coords(0.5, -0.12)
    fig1_14.legend(
        ["IPS", "DM (ニューラルネットワーク)"],
        fontsize=50,
        bbox_to_anchor=(0.5, 1.15),
        ncol=4,
        loc="center",
    )
    fig1_14
    return ax_list_14, fig1_14, i, y


@app.cell
def __(mo):
    mo.md(r"""## 図1.18""")
    return


@app.cell
def __(n_def_actions_ratio_list, plt, result_df, sns, y_label_dict):
    fig1_18, ax_list_18 = plt.subplots(1, 3, figsize=(35, 10), tight_layout=True)
    for i, y in enumerate(["se", "bias", "variance"]):
        ax = ax_list_18[i]
        sns.lineplot(
            markers=True,
            markersize=40,
            linewidth=5,
            legend=False,
            style="est",
            x="n_def_actions_ratio",
            y=y,
            hue="est",
            ax=ax,
            ci=None,
            data=result_df,
        )
        ax.set_title(y_label_dict[y], fontsize=50)
        # yaxis
        ax.set_ylabel("")
        ax.set_ylim(0.0, 0.065)
        ax.tick_params(axis="y", labelsize=30)
        ax.set_yticks([0.0, 0.03, 0.06])
        ax.yaxis.set_label_coords(-0.1, 0.5)
        # xaxis
        if i == 1:
            ax.set_xlabel(r"共通サポートの仮定が満たされない行動の割合$|U|/|A|$", fontsize=50)
        else:
            ax.set_xlabel("", fontsize=40)
        ax.set_xticks(n_def_actions_ratio_list)
        ax.set_xticklabels(n_def_actions_ratio_list, fontsize=35)
        ax.xaxis.set_label_coords(0.5, -0.12)
    fig1_18.legend(
        ["IPS", "DR", "DM (ニューラルネットワーク)"],
        fontsize=50,
        bbox_to_anchor=(0.5, 1.15),
        ncol=4,
        loc="center",
    )
    fig1_18
    return ax_list_18, fig1_18


@app.cell
def __():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
