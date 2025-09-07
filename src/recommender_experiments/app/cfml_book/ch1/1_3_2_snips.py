import marimo

__generated_with = "0.9.24"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    mo.md(r"""# 1.3.2 Self-Normalized Inverse Propensity Score(SNIPS)推定量
    参考文献
    - Adith Swaminathan and Thorsten Joachims. [The Self-Normalized Estimator for Counterfactual Learning](https://papers.nips.cc/paper_files/paper/2015/hash/39027dfad5138c9ca0c474d71db915c3-Abstract.html). NeurIPS2015.""")
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
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.utils import check_random_state
    from tqdm import tqdm
    plt.style.use('ggplot')
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
        DoublyRobust as DR,
    )
    from obp.ope import (
        InverseProbabilityWeighting as IPS,
    )
    from obp.ope import (
        OffPolicyEvaluation,
        RegressionModel,
    )
    from obp.ope import (
        SelfNormalizedInverseProbabilityWeighting as SNIPS,
    )
    from utils import aggregate_simulation_results, eps_greedy_policy
    return (
        DR,
        DataFrame,
        IPS,
        MLPClassifier,
        MLPRegressor,
        OffPolicyEvaluation,
        RegressionModel,
        SNIPS,
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
    mo.md(r"""### (データ収集方策が収集した)ログデータのサイズ$n$を変化させたときのSNIPS推定量の平均二乗誤差・二乗バイアス・バリアンスの挙動""")
    return


@app.cell
def __(check_random_state):
    ## シミュレーション設定
    num_runs = 500  # シミュレーションの繰り返し回数
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
    DR,
    IPS,
    MLPRegressor,
    OffPolicyEvaluation,
    RegressionModel,
    SNIPS,
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
            offline_logged_data = dataset.obtain_batch_bandit_feedback(
                n_rounds=num_data
            )

            ## ログデータ上における評価方策の行動選択確率を計算
            pi = eps_greedy_policy(offline_logged_data["expected_reward"])

            ## 期待報酬関数に対する推定モデル\hat{q}(x,a)を得る
            reg_model = RegressionModel(
                n_actions=dataset.n_actions,
                base_model=MLPRegressor(
                    hidden_layer_sizes=(10, 10), random_state=random_state, max_iter=1000
                ),
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
                    SNIPS(estimator_name="SNIPS"),
                ]
            )
            estimated_policy_values = ope.estimate_policy_values(
                action_dist=pi,
                estimated_rewards_by_reg_model=estimated_rewards_mlp,
            )
            estimated_policy_value_list.append(estimated_policy_values)

        ## シミュレーション結果を集計する
        result_df_list.append(
            aggregate_simulation_results(
                estimated_policy_value_list, policy_value, "num_data", num_data,
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
    mo.md(r"""## 図1.23""")
    return


@app.cell
def __(num_data_list, plt, result_df, sns, y_label_dict):
    fig1_23, ax_list_23 = plt.subplots(1, 3, figsize=(35, 10), tight_layout=True)
    for i23, y23 in enumerate(["se", "bias", "variance"]):
        ax23 = ax_list_23[i23]
        sns.lineplot(
            markers=True,
            markersize=40,
            linewidth=5,
            legend=False,
            style="est",
            x="num_data",
            y=y23,
            hue="est",
            ax=ax23,
            ci=None,
            data=result_df,
        )
        ax23.set_title(y_label_dict[y23], fontsize=50)
        # yaxis
        ax23.set_ylabel("")
        ax23.set_ylim(0.0, 0.53)
        ax23.tick_params(axis="y", labelsize=30)
        ax23.set_yticks([0.0, 0.25, 0.5])
        ax23.yaxis.set_label_coords(-0.1, 0.5)
        # xaxis
        ax23.set_xscale("log")
        if i23 == 1:
            ax23.set_xlabel(r"ログデータのサイズ$n$", fontsize=50)
        else:
            ax23.set_xlabel(r"", fontsize=40)
        ax23.set_xticks(num_data_list)
        ax23.set_xticklabels(num_data_list, fontsize=30)
        ax23.xaxis.set_label_coords(0.5, -0.12)
    fig1_23.legend(
        ["IPS", "DR", "SNIPS"], fontsize=50, bbox_to_anchor=(0.5, 1.15), ncol=4, loc="center",
    )
    fig1_23
    return ax23, ax_list_23, fig1_23, i23, y23


@app.cell
def __():
    import marimo as mo
    return mo,


if __name__ == "__main__":
    app.run()