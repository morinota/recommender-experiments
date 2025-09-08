import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import warnings

    warnings.filterwarnings("ignore")

    import japanize_matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import polars as pl
    import seaborn as sns
    from sklearn.utils import check_random_state
    from tqdm import tqdm

    plt.style.use("ggplot")

    # import open bandit pipeline (obp)
    import obp
    from obp.dataset import SyntheticBanditDatasetWithActionEmbeds as SyntheticBanditDataset
    from obp.dataset import linear_reward_function
    from obp.ope import (
        InverseProbabilityWeighting as IPS,
    )
    from obp.ope import (
        OffPolicyEvaluation,
    )
    from obp.utils import softmax

    return (
        IPS,
        OffPolicyEvaluation,
        SyntheticBanditDataset,
        check_random_state,
        linear_reward_function,
        np,
        obp,
        pl,
        plt,
        sns,
        softmax,
        tqdm,
    )


@app.cell
def _(obp):
    print(f"OBP version: {obp.__version__}")
    return


@app.cell
def _(mo):
    mo.md("""## 1.2.1 オンライン実験による方策性能推定""")
    return


@app.cell
def _(mo):
    mo.md("""### オンライン実験で収集するデータのサイズを変化させたときのAVG推定量の平均二乗誤差の挙動""")
    return


@app.cell
def _(check_random_state):
    # シミュレーション設定
    num_runs = 50  # シミュレーションの繰り返し回数
    dim_context = 10  # 特徴量xの次元
    n_actions = 20  # 行動数, |A|
    beta = 3  # 方策パラメータ
    reward_std = 2  # 報酬のノイズの大きさ
    test_data_size = 100000  # 評価方策の真の性能を近似するためのテストデータのサイズ
    random_state = 12345
    random_ = check_random_state(random_state)
    num_data_list = [250, 500, 1000, 2000, 4000, 8000]  # オンライン実験で収集するデータのサイズ
    return (
        beta,
        dim_context,
        n_actions,
        num_data_list,
        num_runs,
        random_,
        random_state,
        reward_std,
        test_data_size,
    )


@app.cell
def _(
    IPS,
    OffPolicyEvaluation,
    SyntheticBanditDataset,
    beta,
    dim_context,
    linear_reward_function,
    n_actions,
    np,
    num_data_list,
    num_runs,
    pl,
    random_,
    random_state,
    reward_std,
    softmax,
    test_data_size,
    tqdm,
):
    se_df_list_datasize = []  # 各シミュレーション設定での推定誤差(二乗誤差)を格納するリスト

    for num_data in num_data_list:
        ## 合成データ生成クラス
        dataset = SyntheticBanditDataset(
            n_actions=n_actions,
            dim_context=dim_context,
            action_context=random_.normal(size=(n_actions, 10)),  # 行動の特徴量
            beta=beta,
            reward_type="continuous",
            reward_std=reward_std,
            reward_function=linear_reward_function,
            random_state=random_state,
        )

        ## 評価方策の真の性能(policy value)を近似するためのテストデータを生成
        test_data = dataset.obtain_batch_bandit_feedback(n_rounds=test_data_size)

        ## 評価方策の真の性能(policy value)を近似
        value_of_pi = dataset.calc_ground_truth_policy_value(
            expected_reward=test_data["expected_reward"],
            action_dist=softmax(beta * test_data["expected_reward"])[:, :, np.newaxis],
        )

        se_list = []  # 単一のシミュレーション設定での、試行回数分の推定誤差(二乗誤差)を格納するリスト
        for _ in tqdm(range(num_runs), desc=f"num_data: {num_data}..."):
            ## オンライン実験で収集するログデータを、評価方策が形成する同時分布に従い生成
            online_experiment_data = dataset.obtain_batch_bandit_feedback(n_rounds=num_data)

            ## ログデータ上における評価方策の行動選択確率を計算
            pi = softmax(beta * online_experiment_data["expected_reward"])

            ## オンライン実験で収集したログデータを用いて、評価方策の性能(policy value)の推定を実行する
            ope = OffPolicyEvaluation(
                bandit_feedback=online_experiment_data,
                ope_estimators=[
                    IPS(estimator_name="AVG")
                ],  # IPS推定量は、オンライン実験の設定においてはAVG推定量に一致
            )

            squared_errors = ope.evaluate_performance_of_estimators(
                ground_truth_policy_value=value_of_pi,  # V(\pi)
                action_dist=pi[:, :, np.newaxis],  # \pi(a|x)
                metric="se",  # 推定誤差のmetricとして二乗誤差(se: squared error)を指定
            )
            se_list.append(squared_errors)

        ## シミュレーション結果を集計
        se_df = pl.DataFrame(se_list).select(
            pl.lit(num_data).alias("num_data"),
            pl.col("AVG").alias("se"),  # 推定誤差(二乗誤差)
        )
        se_df_list_datasize.append(se_df)

    result_df_datasize = pl.concat(se_df_list_datasize)
    print("データサイズ変化実験の結果:")
    print(result_df_datasize)
    return (result_df_datasize,)


@app.cell
def _(mo):
    mo.md("""### 報酬のノイズの大きさを変化させたときのAVG推定量の平均二乗誤差の挙動""")
    return


@app.cell
def _():
    # 実験設定
    num_data_noise = 1000  # オンライン実験で収集するデータのサイズ
    noise_list = [0, 2, 4, 6, 8, 10]  # 報酬のノイズの大きさ
    return noise_list, num_data_noise


@app.cell
def _(
    IPS,
    OffPolicyEvaluation,
    SyntheticBanditDataset,
    beta,
    dim_context,
    linear_reward_function,
    n_actions,
    noise_list,
    np,
    num_data_noise,
    num_runs,
    pl,
    random_,
    random_state,
    softmax,
    test_data_size,
    tqdm,
):
    se_df_list_noise = []  # 各シミュレーション設定での推定誤差(二乗誤差)を格納するリスト

    for noise in noise_list:
        ## 合成データ生成クラス
        dataset_noise = SyntheticBanditDataset(
            n_actions=n_actions,
            dim_context=dim_context,
            action_context=random_.normal(size=(n_actions, 10)),  # 行動の特徴量
            beta=beta,
            reward_type="continuous",
            reward_std=noise,
            reward_function=linear_reward_function,
            random_state=random_state,
        )

        ## 評価方策の真の性能(policy value)を近似するためのテストデータを生成
        test_data_noise = dataset_noise.obtain_batch_bandit_feedback(n_rounds=test_data_size)

        ## 評価方策の真の性能(policy value)を近似
        value_of_pi_noise = dataset_noise.calc_ground_truth_policy_value(
            expected_reward=test_data_noise["expected_reward"],
            action_dist=softmax(beta * test_data_noise["expected_reward"])[:, :, np.newaxis],
        )

        se_list_noise = []  # 単一のシミュレーション設定での、試行回数分の推定誤差(二乗誤差)を格納するリスト
        for _ in tqdm(range(num_runs), desc=f"noise: {noise}..."):
            ## オンライン実験で収集するログデータを、評価方策が形成する同時分布に従い生成
            online_experiment_data_noise = dataset_noise.obtain_batch_bandit_feedback(n_rounds=num_data_noise)

            ## ログデータ上における評価方策の行動選択確率を計算
            pi_noise = softmax(beta * online_experiment_data_noise["expected_reward"])

            ## オンライン実験で収集したログデータを用いて、評価方策の性能(policy value)の推定を実行する
            ope_noise = OffPolicyEvaluation(
                bandit_feedback=online_experiment_data_noise,
                ope_estimators=[
                    IPS(estimator_name="AVG")
                ],  # IPS推定量は、オンライン実験の設定においてはAVG推定量に一致
            )

            squared_errors_noise = ope_noise.evaluate_performance_of_estimators(
                ground_truth_policy_value=value_of_pi_noise,  # V(\pi)
                action_dist=pi_noise[:, :, np.newaxis],  # \pi(a|x)
                metric="se",  # 推定誤差のmetricとして二乗誤差(se: squared error)を指定
            )
            se_list_noise.append(squared_errors_noise)

        ## シミュレーション結果を集計
        se_df_noise = pl.DataFrame(se_list_noise).select(
            pl.lit(noise).alias("reward_std"),
            pl.col("AVG").alias("se"),  # 推定誤差(二乗誤差)
        )
        se_df_list_noise.append(se_df_noise)

    result_df_noise = pl.concat(se_df_list_noise)
    print("ノイズ変化実験の結果:")
    print(result_df_noise)
    return (result_df_noise,)


@app.cell
def _(mo):
    mo.md("""### 図1.8: 平均二乗誤差の可視化""")
    return


@app.cell
def _(pl, plt, result_df_datasize, result_df_noise, sns):
    fig, ax_list = plt.subplots(1, 2, figsize=(25, 9), tight_layout=True)
    x_dict = {0: "num_data", 1: "reward_std"}
    title_dict = {0: "左図", 1: "右図"}

    for i, result_df in enumerate([result_df_datasize, result_df_noise]):
        ax = ax_list[i]

        # グループごとのMSEを計算
        mse_df = result_df.group_by(x_dict[i]).agg(pl.col("se").mean().alias("se")).to_pandas()

        sns.lineplot(
            markers=True,  # マーカーを表示
            markersize=35,  # マーカーのサイズ
            linewidth=9,
            legend=False,
            x=x_dict[i],
            y="se",
            ax=ax,
            data=mse_df,
        )

        ax.set_title(title_dict[i], fontsize=50)
        if i == 0:
            ax.set_ylabel("平均二乗誤差 (MSE)", fontsize=40)
            ax.set_yticks([0.01, 0.02, 0.03])
            ax.set_yticklabels([0.01, 0.02, 0.03], fontsize=26)
            ax.yaxis.set_label_coords(-0.11, 0.5)
            ax.set_xlabel(r"オンライン実験で収集したデータ数$n$", fontsize=45)
            ax.set_xticks([250, 2000, 4000, 8000])
            ax.set_xticklabels([250, 2000, 4000, 8000], fontsize=30)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelsize=26)
            ax.set_xlabel(r"報酬ノイズの大きさ$\sigma(x,a)$", fontsize=45)
            ax.set_xticks([0, 2, 4, 6, 8, 10])
            ax.set_xticklabels([0, 2, 4, 6, 8, 10], fontsize=30)
        ax.xaxis.set_label_coords(0.5, -0.12)

    plt.show()
    return


if __name__ == "__main__":
    app.run()
