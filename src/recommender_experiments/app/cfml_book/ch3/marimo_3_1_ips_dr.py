import marimo

__generated_with = "0.9.24"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    mo.md(r"""# 3.1 困難な状況におけるIPS・DR推定量の精度悪化
    参考文献
    - Yuta Saito and Thorsten Joachims. [Off-Policy Evaluation for Large Action Spaces via Embeddings](https://arxiv.org/abs/2202.06317). ICML2022.""")
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
    y_label_dict = {"se": "左図：平均二乗誤差", "bias": "中図：二乗バイアス", "variance": "右図：バリアンス"}

    from dataset import calc_true_value, generate_synthetic_data
    from estimators import calc_avg, calc_dr, calc_ips
    from recommender_experiments.app.cfml_book.common_utils import eps_greedy_policy
    from utils import aggregate_simulation_results

    return (
        DataFrame,
        aggregate_simulation_results,
        calc_avg,
        calc_dr,
        calc_ips,
        calc_true_value,
        check_random_state,
        eps_greedy_policy,
        generate_synthetic_data,
        japanize_matplotlib,
        np,
        pd,
        plt,
        sns,
        tqdm,
        warnings,
        y_label_dict,
    )


@app.cell
def __(mo):
    mo.md(r"""### 行動数$|A|$を変化させたときのAVG・IPS・DR推定量の平均二乗誤差・二乗バイアス・バリアンスの挙動""")
    return


@app.cell
def __(check_random_state):
    ## シミュレーション設定
    num_runs = 50  # シミュレーションの繰り返し回数
    dim_context = 5  # 特徴量xの次元
    num_clusters = 30  # 行動クラスタ数
    num_data_actions = 500  # ログデータのサイズ
    beta_actions = -0.2  # データ収集方策のパラメータ
    lambda_actions = 0.0  # クラスタ効果と残差効果の配合率
    random_state_actions = 12345
    random_actions = check_random_state(random_state_actions)
    num_actions_list = [250, 500, 1000, 2000, 4000]  # 行動数, |A|
    return (
        beta_actions,
        dim_context,
        lambda_actions,
        num_actions_list,
        num_clusters,
        num_data_actions,
        num_runs,
        random_actions,
        random_state_actions,
    )


@app.cell
def __(
    aggregate_simulation_results,
    beta_actions,
    calc_avg,
    calc_dr,
    calc_ips,
    calc_true_value,
    dim_context,
    eps_greedy_policy,
    generate_synthetic_data,
    lambda_actions,
    num_actions_list,
    num_clusters,
    num_data_actions,
    num_runs,
    pd,
    random_actions,
    tqdm,
):
    result_df_list_actions = []
    theta_g_actions = random_actions.normal(size=(dim_context, num_clusters))
    M_g_actions = random_actions.normal(size=(dim_context, num_clusters))
    b_g_actions = random_actions.normal(size=(1, num_clusters))
    for num_actions in num_actions_list:
        ## 期待報酬関数を定義するためのパラメータを抽出
        phi_a_actions = random_actions.choice(num_clusters, size=num_actions)
        theta_h_actions = random_actions.normal(size=(dim_context, num_actions))
        M_h_actions = random_actions.normal(size=(dim_context, num_actions))
        b_h_actions = random_actions.normal(size=(1, num_actions))

        ## 評価方策の真の性能(policy value)を計算
        policy_value_actions = calc_true_value(
            dim_context=dim_context,
            num_actions=num_actions,
            num_clusters=num_clusters,
            lambda_=lambda_actions,
            phi_a=phi_a_actions,
            theta_g=theta_g_actions,
            M_g=M_g_actions,
            b_g=b_g_actions,
            theta_h=theta_h_actions,
            M_h=M_h_actions,
            b_h=b_h_actions,
        )

        estimated_policy_value_list_actions = []
        for _ in tqdm(range(num_runs), desc=f"num_actions={num_actions}..."):
            ## データ収集方策が形成する分布に従いログデータを生成
            offline_logged_data_actions = generate_synthetic_data(
                num_data=num_data_actions,
                lambda_=lambda_actions,
                beta=beta_actions,
                theta_g=theta_g_actions,
                M_g=M_g_actions,
                b_g=b_g_actions,
                theta_h=theta_h_actions,
                M_h=M_h_actions,
                b_h=b_h_actions,
                phi_a=phi_a_actions,
                dim_context=dim_context,
                num_actions=num_actions,
                num_clusters=num_clusters,
                random_state=_,
            )

            ## ログデータ上における評価方策の行動選択確率を計算
            pi_actions = eps_greedy_policy(offline_logged_data_actions["q_x_a"], k=5, eps=0.1, return_normalized=True, rank_method="ordinal")

            ## ログデータを用いてオフ方策評価を実行する
            estimated_policy_values_actions = dict()
            estimated_policy_values_actions["avg"] = calc_avg(offline_logged_data_actions)
            estimated_policy_values_actions["ips"] = calc_ips(offline_logged_data_actions, pi_actions)
            estimated_policy_values_actions["dr"] = calc_dr(
                offline_logged_data_actions,
                pi_actions,
                offline_logged_data_actions["q_x_a"]
                + random_actions.normal(-1, scale=1 / 2, size=(num_data_actions, num_actions)),
            )
            estimated_policy_value_list_actions.append(estimated_policy_values_actions)

        ## シミュレーション結果を集計する
        result_df_list_actions.append(
            aggregate_simulation_results(
                estimated_policy_value_list_actions,
                policy_value_actions,
                "num_actions",
                num_actions,
            )
        )
    result_df_actions = pd.concat(result_df_list_actions).reset_index(level=0)
    return (
        M_g_actions,
        M_h_actions,
        b_g_actions,
        b_h_actions,
        estimated_policy_value_list_actions,
        estimated_policy_values_actions,
        num_actions,
        offline_logged_data_actions,
        phi_a_actions,
        pi_actions,
        policy_value_actions,
        result_df_actions,
        result_df_list_actions,
        theta_g_actions,
        theta_h_actions,
    )


@app.cell
def __(mo):
    mo.md(r"""## 図3.1""")
    return


@app.cell
def __(num_actions_list, plt, result_df_actions, sns, y_label_dict):
    fig3_1, ax_list_1 = plt.subplots(1, 3, figsize=(35, 10), tight_layout=True)
    for i1, y1 in enumerate(["se", "bias", "variance"]):
        ax1 = ax_list_1[i1]
        sns.lineplot(
            markers=True,
            markersize=40,
            linewidth=7,
            legend=False,
            style="est",
            x="num_actions",
            y=y1,
            hue="est",
            ax=ax1,
            ci=None,
            palette=["tab:grey", "tab:red", "tab:blue"],
            data=result_df_actions,
        )
        ax1.set_title(y_label_dict[y1], fontsize=50)
        # yaxis
        ax1.set_ylabel("")
        ax1.set_ylim(0.0, 15.5)
        ax1.tick_params(axis="y", labelsize=30)
        ax1.yaxis.set_label_coords(-0.1, 0.5)
        ax1.set_yticks([0, 2, 4, 6, 8, 10, 12, 14])
        # xaxis
        ax1.set_xscale("log")
        if i1 == 1:
            ax1.set_xlabel(r"行動の数$|\mathcal{A}|$", fontsize=50)
        else:
            ax1.set_xlabel(r"", fontsize=40)
        ax1.set_xticks(num_actions_list)
        ax1.set_xticklabels(num_actions_list, fontsize=30)
        ax1.xaxis.set_label_coords(0.5, -0.12)
    fig3_1.legend(["AVG", "IPS", "DR"], fontsize=50, bbox_to_anchor=(0.5, 1.15), ncol=4, loc="center")
    fig3_1
    return ax1, ax_list_1, fig3_1, i1, y1


@app.cell
def __(mo):
    mo.md(
        r"""### 行動数が多い状況で、共通サポートが満たされない行動の割合$|U|/|A|$を変化させたときのAVG・IPS・DR推定量の平均二乗誤差・二乗バイアス・バリアンスの挙動"""
    )
    return


@app.cell
def __(check_random_state):
    ## シミュレーション設定
    num_runs_def = 50  # シミュレーションの繰り返し回数
    dim_context_def = 5  # 特徴量xの次元
    num_actions_def = 1000  # 行動数, |A|
    num_clusters_def = 30  # 行動クラスタ数
    num_data_def = 1000  # ログデータのサイズ
    beta_def = 1  # データ収集方策のパラメータ
    lambda_def = 0.0  # クラスタ効果と残差効果の配合率
    random_state_def = 12345
    random_def = check_random_state(random_state_def)
    num_def_actions_list = [0, 100, 200, 300, 400]  # 共通サポートが満たされない行動の数
    return (
        beta_def,
        dim_context_def,
        lambda_def,
        num_actions_def,
        num_clusters_def,
        num_data_def,
        num_def_actions_list,
        num_runs_def,
        random_def,
        random_state_def,
    )


@app.cell
def __(
    aggregate_simulation_results,
    beta_def,
    calc_avg,
    calc_dr,
    calc_ips,
    calc_true_value,
    dim_context_def,
    eps_greedy_policy,
    generate_synthetic_data,
    lambda_def,
    num_actions_def,
    num_clusters_def,
    num_data_def,
    num_def_actions_list,
    num_runs_def,
    pd,
    random_def,
    tqdm,
):
    result_df_list_def = []
    theta_g_def = random_def.normal(size=(dim_context_def, num_clusters_def))
    M_g_def = random_def.normal(size=(dim_context_def, num_clusters_def))
    b_g_def = random_def.normal(size=(1, num_clusters_def))
    for num_def_actions in num_def_actions_list:
        ## 期待報酬関数を定義するためのパラメータを抽出
        phi_a_def = random_def.choice(num_clusters_def, size=num_actions_def)
        theta_h_def = random_def.normal(size=(dim_context_def, num_actions_def))
        M_h_def = random_def.normal(size=(dim_context_def, num_actions_def))
        b_h_def = random_def.normal(size=(1, num_actions_def))

        ## 評価方策の真の性能(policy value)を計算
        policy_value_def = calc_true_value(
            dim_context=dim_context_def,
            num_actions=num_actions_def,
            num_clusters=num_clusters_def,
            lambda_=lambda_def,
            phi_a=phi_a_def,
            theta_g=theta_g_def,
            M_g=M_g_def,
            b_g=b_g_def,
            theta_h=theta_h_def,
            M_h=M_h_def,
            b_h=b_h_def,
        )

        estimated_policy_value_list_def = []
        for _ in tqdm(range(num_runs_def), desc=f"num_def_actions={num_def_actions}..."):
            ## データ収集方策が形成する分布に従いログデータを生成
            offline_logged_data_def = generate_synthetic_data(
                num_data=num_data_def,
                lambda_=lambda_def,
                beta=beta_def,
                theta_g=theta_g_def,
                M_g=M_g_def,
                b_g=b_g_def,
                theta_h=theta_h_def,
                M_h=M_h_def,
                b_h=b_h_def,
                phi_a=phi_a_def,
                dim_context=dim_context_def,
                num_actions=num_actions_def,
                num_def_actions=num_def_actions,
                num_clusters=num_clusters_def,
                random_state=_,
            )

            ## ログデータ上における評価方策の行動選択確率を計算
            pi_def = eps_greedy_policy(offline_logged_data_def["q_x_a"], k=5, eps=0.1, return_normalized=True, rank_method="ordinal")

            ## ログデータを用いてオフ方策評価を実行する
            estimated_policy_values_def = dict()
            estimated_policy_values_def["avg"] = calc_avg(offline_logged_data_def)
            estimated_policy_values_def["ips"] = calc_ips(offline_logged_data_def, pi_def)
            estimated_policy_values_def["dr"] = calc_dr(
                offline_logged_data_def,
                pi_def,
                offline_logged_data_def["q_x_a"]
                + random_def.normal(-1, scale=1 / 2, size=(num_data_def, num_actions_def)),
            )
            estimated_policy_value_list_def.append(estimated_policy_values_def)

        ## シミュレーション結果を集計する
        result_df_list_def.append(
            aggregate_simulation_results(
                estimated_policy_value_list_def,
                policy_value_def,
                "num_def_actions",
                num_def_actions / num_actions_def,
            )
        )
    result_df_def_actions = pd.concat(result_df_list_def).reset_index(level=0)
    return (
        M_g_def,
        M_h_def,
        b_g_def,
        b_h_def,
        estimated_policy_value_list_def,
        estimated_policy_values_def,
        num_def_actions,
        offline_logged_data_def,
        phi_a_def,
        pi_def,
        policy_value_def,
        result_df_def_actions,
        result_df_list_def,
        theta_g_def,
        theta_h_def,
    )


@app.cell
def __(mo):
    mo.md(r"""## 図3.3""")
    return


@app.cell
def __(plt, result_df_def_actions, sns, y_label_dict):
    fig3_3, ax_list_3 = plt.subplots(1, 3, figsize=(35, 10), tight_layout=True)
    for i3, y3 in enumerate(["se", "bias", "variance"]):
        ax3 = ax_list_3[i3]
        sns.lineplot(
            markers=True,
            markersize=40,
            linewidth=7,
            legend=False,
            style="est",
            x="num_def_actions",
            y=y3,
            hue="est",
            ax=ax3,
            ci=None,
            palette=["tab:grey", "tab:red", "tab:blue"],
            data=result_df_def_actions,
        )
        ax3.set_title(y_label_dict[y3], fontsize=50)
        # yaxis
        ax3.set_ylabel("")
        ax3.set_ylim(0.0, 2)
        ax3.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
        ax3.tick_params(axis="y", labelsize=30)
        ax3.yaxis.set_label_coords(-0.1, 0.5)
        # xaxis
        if i3 == 1:
            ax3.set_xlabel(r"共通サポートの仮定が満たされない行動の割合$|U|/|A|$", fontsize=50)
        else:
            ax3.set_xlabel(r"", fontsize=40)
        ax3.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4])
        ax3.set_xticklabels([0.0, 0.1, 0.2, 0.3, 0.4], fontsize=32)
        ax3.xaxis.set_label_coords(0.5, -0.12)
    fig3_3.legend(["AVG", "IPS", "DR"], fontsize=50, bbox_to_anchor=(0.5, 1.15), ncol=4, loc="center")
    fig3_3
    return ax3, ax_list_3, fig3_3, i3, y3


@app.cell
def __(mo):
    mo.md(
        r"""### 行動数が多い状況で、ログデータのサイズ$n$を変化させたときのAVG・IPS・DR推定量の平均二乗誤差・二乗バイアス・バリアンスの挙動"""
    )
    return


@app.cell
def __(check_random_state):
    ## シミュレーション設定
    num_runs_data = 50  # シミュレーションの繰り返し回数
    dim_context_data = 5  # 特徴量xの次元
    num_actions_data = 1000  # 行動数, |A|
    num_clusters_data = 30  # 行動クラスタ数
    beta_data = -0.2  # データ収集方策のパラメータ
    lambda_data = 0.0  # クラスタ効果と残差効果の配合率
    random_state_data = 12345
    random_data = check_random_state(random_state_data)
    num_data_list = [250, 500, 1000, 2000, 4000]  # ログデータのサイズ
    return (
        beta_data,
        dim_context_data,
        lambda_data,
        num_actions_data,
        num_clusters_data,
        num_data_list,
        num_runs_data,
        random_data,
        random_state_data,
    )


@app.cell
def __(
    aggregate_simulation_results,
    beta_data,
    calc_avg,
    calc_dr,
    calc_ips,
    calc_true_value,
    dim_context_data,
    eps_greedy_policy,
    generate_synthetic_data,
    lambda_data,
    num_actions_data,
    num_clusters_data,
    num_data_list,
    num_runs_data,
    pd,
    random_data,
    tqdm,
):
    result_df_list_data = []
    ## 期待報酬関数を定義するためのパラメータを抽出
    phi_a_data = random_data.choice(num_clusters_data, size=num_actions_data)
    theta_g_data = random_data.normal(size=(dim_context_data, num_clusters_data))
    M_g_data = random_data.normal(size=(dim_context_data, num_clusters_data))
    b_g_data = random_data.normal(size=(1, num_clusters_data))
    theta_h_data = random_data.normal(size=(dim_context_data, num_actions_data))
    M_h_data = random_data.normal(size=(dim_context_data, num_actions_data))
    b_h_data = random_data.normal(size=(1, num_actions_data))
    for num_data in num_data_list:
        ## 評価方策の真の性能(policy value)を計算
        policy_value_data = calc_true_value(
            dim_context=dim_context_data,
            num_actions=num_actions_data,
            num_clusters=num_clusters_data,
            lambda_=lambda_data,
            phi_a=phi_a_data,
            theta_g=theta_g_data,
            M_g=M_g_data,
            b_g=b_g_data,
            theta_h=theta_h_data,
            M_h=M_h_data,
            b_h=b_h_data,
        )

        estimated_policy_value_list_data = []
        for _ in tqdm(range(num_runs_data), desc=f"num_data={num_data}..."):
            ## データ収集方策が形成する分布に従いログデータを生成
            offline_logged_data_data = generate_synthetic_data(
                num_data=num_data,
                lambda_=lambda_data,
                beta=beta_data,
                theta_g=theta_g_data,
                M_g=M_g_data,
                b_g=b_g_data,
                theta_h=theta_h_data,
                M_h=M_h_data,
                b_h=b_h_data,
                phi_a=phi_a_data,
                dim_context=dim_context_data,
                num_actions=num_actions_data,
                num_clusters=num_clusters_data,
                random_state=_,
            )

            ## ログデータ上における評価方策の行動選択確率を計算
            pi_data = eps_greedy_policy(offline_logged_data_data["q_x_a"], k=5, eps=0.1, return_normalized=True, rank_method="ordinal")

            ## ログデータを用いてオフ方策評価を実行する
            estimated_policy_values_data = dict()
            estimated_policy_values_data["avg"] = calc_avg(offline_logged_data_data)
            estimated_policy_values_data["ips"] = calc_ips(offline_logged_data_data, pi_data)
            estimated_policy_values_data["dr"] = calc_dr(
                offline_logged_data_data,
                pi_data,
                offline_logged_data_data["q_x_a"]
                + random_data.normal(-1, scale=1 / 2, size=(num_data, num_actions_data)),
            )
            estimated_policy_value_list_data.append(estimated_policy_values_data)

        ## シミュレーション結果を集計する
        result_df_list_data.append(
            aggregate_simulation_results(
                estimated_policy_value_list_data,
                policy_value_data,
                "num_data",
                num_data,
            )
        )
    result_df_data = pd.concat(result_df_list_data).reset_index(level=0)
    return (
        M_g_data,
        M_h_data,
        b_g_data,
        b_h_data,
        estimated_policy_value_list_data,
        estimated_policy_values_data,
        num_data,
        offline_logged_data_data,
        phi_a_data,
        pi_data,
        policy_value_data,
        result_df_data,
        result_df_list_data,
        theta_g_data,
        theta_h_data,
    )


@app.cell
def __(mo):
    mo.md(r"""## 図3.4""")
    return


@app.cell
def __(num_data_list, plt, result_df_data, sns, y_label_dict):
    fig3_4, ax_list_4 = plt.subplots(1, 3, figsize=(35, 10), tight_layout=True)
    for i4, y4 in enumerate(["se", "bias", "variance"]):
        ax4 = ax_list_4[i4]
        sns.lineplot(
            markers=True,
            markersize=40,
            linewidth=7,
            legend=False,
            style="est",
            x="num_data",
            y=y4,
            hue="est",
            ax=ax4,
            ci=None,
            palette=["tab:grey", "tab:red", "tab:blue"],
            data=result_df_data,
        )
        ax4.set_title(y_label_dict[y4], fontsize=50)
        # yaxis
        ax4.set_ylabel("")
        ax4.set_ylim(0.0, 10.5)
        ax4.set_yticks([0.0, 2, 4, 6, 8, 10])
        ax4.tick_params(axis="y", labelsize=30)
        ax4.yaxis.set_label_coords(-0.1, 0.5)
        # xaxis
        ax4.set_xscale("log")
        if i4 == 1:
            ax4.set_xlabel(r"ログデータのサイズ$n$", fontsize=50)
        else:
            ax4.set_xlabel(r"", fontsize=40)
        ax4.set_xticks(num_data_list)
        ax4.set_xticklabels(num_data_list, fontsize=30)
        ax4.xaxis.set_label_coords(0.5, -0.12)
    fig3_4.legend(["AVG", "IPS", "DR"], fontsize=50, bbox_to_anchor=(0.5, 1.15), ncol=4, loc="center")
    fig3_4
    return ax4, ax_list_4, fig3_4, i4, y4


@app.cell
def __():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
