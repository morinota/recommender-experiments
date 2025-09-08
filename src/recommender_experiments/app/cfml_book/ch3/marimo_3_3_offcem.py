import marimo

__generated_with = "0.10.19"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    mo.md(
        r"""
        # 3.3 Off-Policy Estimator based on the Conjunct Effect Model(OffCEM)推定量
        参考文献
        - Yuta Saito, Qingyang Ren, and Thorsten Joachims. [Off-Policy Evaluation for Large Action Spaces via Conjunct Effect Modeling](https://arxiv.org/abs/2305.08062). ICML2023.
        """
    )
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
    from estimators import calc_ips, calc_mips, calc_offcem
    from utils import aggregate_simulation_results, eps_greedy_policy, remove_outliers

    return (
        DataFrame,
        aggregate_simulation_results,
        calc_ips,
        calc_mips,
        calc_offcem,
        calc_true_value,
        check_random_state,
        eps_greedy_policy,
        generate_synthetic_data,
        japanize_matplotlib,
        np,
        pd,
        plt,
        remove_outliers,
        sns,
        tqdm,
        warnings,
        y_label_dict,
    )


@app.cell
def __(mo):
    mo.md(
        r"""### あえて用いない行動特徴量の次元数を変化させたときのIPS・MIPS・OffCEM推定量の平均二乗誤差・二乗バイアス・バリアンスの挙動"""
    )
    return


@app.cell
def __(check_random_state):
    ## シミュレーション設定
    num_runs = 50  # シミュレーションの繰り返し回数
    dim_context = 5  # 特徴量xの次元
    num_actions = 1000  # 行動数, |A|
    num_clusters = 990  # 行動クラスタ数
    num_data = 1000  # ログデータのサイズ
    random_state = 12345
    lambda_ = 0.5  # クラスタ効果と残差効果の配合率
    beta = -0.1  # データ収集方策のパラメータ
    random_ = check_random_state(random_state)
    num_replace_clusters_list = [0, 100, 300, 500, 700, 900]  # あえて用いない行動特徴量の次元数
    return (
        beta,
        dim_context,
        lambda_,
        num_actions,
        num_clusters,
        num_data,
        num_replace_clusters_list,
        num_runs,
        random_,
        random_state,
    )


@app.cell
def __(
    aggregate_simulation_results,
    beta,
    calc_ips,
    calc_mips,
    calc_offcem,
    calc_true_value,
    dim_context,
    eps_greedy_policy,
    generate_synthetic_data,
    lambda_,
    np,
    num_actions,
    num_clusters,
    num_data,
    num_replace_clusters_list,
    num_runs,
    pd,
    random_,
    tqdm,
):
    result_df_list = []
    ## 期待報酬関数を定義するためのパラメータを抽出
    phi_a = np.tile(np.arange(num_clusters), 100)[:num_actions]
    theta_g = random_.normal(size=(dim_context, num_clusters))
    M_g = random_.normal(size=(dim_context, num_clusters))
    b_g = random_.normal(size=(1, num_clusters))
    theta_h = random_.normal(size=(dim_context, num_actions))
    M_h = random_.normal(size=(dim_context, num_actions))
    b_h = random_.normal(size=(1, num_actions))
    for num_replace_clusters in num_replace_clusters_list:
        ## 評価方策の真の性能(policy value)を計算
        policy_value = calc_true_value(
            dim_context=dim_context,
            num_actions=num_actions,
            num_clusters=num_clusters,
            lambda_=lambda_,
            phi_a=phi_a,
            theta_g=theta_g,
            M_g=M_g,
            b_g=b_g,
            theta_h=theta_h,
            M_h=M_h,
            b_h=b_h,
        )

        estimated_policy_value_list = []
        for _ in tqdm(range(num_runs), desc=f"num_replace_clusters={num_replace_clusters}..."):
            ## データ収集方策が形成する分布に従いログデータを生成
            offline_logged_data = generate_synthetic_data(
                num_data=num_data,
                lambda_=lambda_,
                beta=beta,
                theta_g=theta_g,
                M_g=M_g,
                b_g=b_g,
                theta_h=theta_h,
                M_h=M_h,
                b_h=b_h,
                phi_a=phi_a,
                dim_context=dim_context,
                num_actions=num_actions,
                num_clusters=num_clusters,
                random_state=_,
            )

            ## ログデータ上における評価方策の行動選択確率を計算
            pi = eps_greedy_policy(offline_logged_data["q_x_a"])

            ## ログデータを用いてオフ方策評価を実行する
            estimated_policy_values = dict()
            estimated_policy_values["ips"] = calc_ips(offline_logged_data, pi)
            estimated_policy_values["mips"] = calc_mips(offline_logged_data, pi, replace_c=num_replace_clusters)
            q_hat = offline_logged_data["h_x_a"] + random_.normal(size=(num_data, num_actions))
            estimated_policy_values["offcem"] = calc_offcem(
                offline_logged_data, pi, q_hat, replace_c=num_replace_clusters
            )
            estimated_policy_value_list.append(estimated_policy_values)

        ## シミュレーション結果を集計する
        result_df_list.append(
            aggregate_simulation_results(
                estimated_policy_value_list,
                policy_value,
                "num_replace_clusters",
                num_replace_clusters,
            )
        )
    result_df_num_replace_clusters = pd.concat(result_df_list).reset_index(level=0)
    return (
        M_g,
        M_h,
        b_g,
        b_h,
        estimated_policy_value_list,
        estimated_policy_values,
        num_replace_clusters,
        offline_logged_data,
        phi_a,
        pi,
        policy_value,
        q_hat,
        result_df_list,
        result_df_num_replace_clusters,
        theta_g,
        theta_h,
    )


@app.cell
def __(mo):
    mo.md(r"""## 図3.15""")
    return


@app.cell
def __(num_replace_clusters_list, plt, result_df_num_replace_clusters, sns, y_label_dict):
    fig, ax_list = plt.subplots(1, 3, figsize=(35, 10), tight_layout=True)
    for i, y in enumerate(["se", "bias", "variance"]):
        ax = ax_list[i]
        sns.lineplot(
            markers=True,
            markersize=40,
            linewidth=10,
            legend=False,
            style="est",
            x="num_replace_clusters",
            y=y,
            hue="est",
            ax=ax,
            ci=None,
            palette=["tab:red", "tab:orange"],
            data=result_df_num_replace_clusters.query("est != 'avg' and est != 'offcem'"),
        )
        ax.set_title(y_label_dict[y], fontsize=50)
        # yaxis
        ax.set_ylabel("")
        ax.set_ylim(0.0, 1.8)
        ax.set_yticks([0, 0.5, 1, 1.5])
        ax.tick_params(axis="y", labelsize=30)
        ax.yaxis.set_label_coords(-0.1, 0.5)
        # xaxis
        if i == 1:
            ax.set_xlabel(r"観測されない（もしくはあえて使用しない）行動特徴量の次元数", fontsize=50)
        else:
            ax.set_xlabel(r"", fontsize=40)
        ax.set_xticks(num_replace_clusters_list)
        ax.set_xticklabels(num_replace_clusters_list, fontsize=30)
        ax.xaxis.set_label_coords(0.5, -0.12)
    fig.legend(["IPS", "MIPS"], fontsize=50, bbox_to_anchor=(0.5, 1.15), ncol=4, loc="center")
    plt.show()
    return ax, ax_list, fig, i, y


@app.cell
def __(mo):
    mo.md(r"""## 図3.19""")
    return


@app.cell
def __(num_replace_clusters_list, plt, result_df_num_replace_clusters, sns, y_label_dict):
    fig2, ax_list2 = plt.subplots(1, 3, figsize=(35, 10), tight_layout=True)
    for i2, y2 in enumerate(["se", "bias", "variance"]):
        ax2 = ax_list2[i2]
        sns.lineplot(
            markers=True,
            markersize=40,
            linewidth=10,
            legend=False,
            style="est",
            x="num_replace_clusters",
            y=y2,
            hue="est",
            ax=ax2,
            ci=None,
            palette=["tab:red", "tab:orange", "tab:purple"],
            data=result_df_num_replace_clusters.query("est != 'avg'"),
        )
        ax2.set_title(y_label_dict[y2], fontsize=50)
        # yaxis
        ax2.set_ylabel("")
        ax2.set_ylim(0.0, 1.8)
        ax2.set_yticks([0, 0.5, 1, 1.5])
        ax2.tick_params(axis="y", labelsize=30)
        ax2.yaxis.set_label_coords(-0.1, 0.5)
        # xaxis
        if i2 == 1:
            ax2.set_xlabel(r"観測されない（もしくはあえて使用しない）行動特徴量の次元数", fontsize=50)
        else:
            ax2.set_xlabel(r"", fontsize=40)
        ax2.set_xticks(num_replace_clusters_list)
        ax2.set_xticklabels(num_replace_clusters_list, fontsize=30)
        ax2.xaxis.set_label_coords(0.5, -0.12)
    fig2.legend(["IPS", "MIPS", "OffCEM"], fontsize=50, bbox_to_anchor=(0.5, 1.15), ncol=4, loc="center")
    plt.show()
    return ax2, ax_list2, fig2, i2, y2


@app.cell
def __(mo):
    mo.md(r"""### 行動数$|A|$を変化させたときのIPS・MIPS・OffCEM推定量の平均二乗誤差・二乗バイアス・バリアンスの挙動""")
    return


@app.cell
def __(check_random_state):
    ## シミュレーション設定
    num_runs_4 = 50  # シミュレーションの繰り返し回数
    dim_context_2 = 5  # 特徴量xの次元
    num_data_2 = 1000  # ログデータのサイズ
    beta_2 = -0.2  # データ収集方策のパラメータ
    lambda_2 = 0.8  # クラスタ効果と残差効果の配合率
    random_state_2 = 12345
    random_2 = check_random_state(random_state_2)
    num_actions_list = [250, 500, 1000, 2000, 4000]  # 行動数, |A|
    return (
        beta_2,
        dim_context_2,
        lambda_2,
        num_actions_list,
        num_data_2,
        num_runs_2,
        random_2,
        random_state_2,
    )


@app.cell
def __(
    aggregate_simulation_results,
    beta_2,
    calc_ips,
    calc_mips,
    calc_offcem,
    calc_true_value,
    dim_context_2,
    eps_greedy_policy,
    generate_synthetic_data,
    lambda_2,
    np,
    num_actions_list,
    num_data_2,
    num_runs_2,
    pd,
    random_2,
    tqdm,
):
    result_df_list_2 = []
    for num_actions_item in num_actions_list:
        ## 期待報酬関数を定義するためのパラメータを抽出
        num_clusters_2 = np.int(0.995 * num_actions_item)  # 行動クラスタ数
        theta_g_2 = random_2.normal(size=(dim_context_2, num_clusters_2))
        M_g_2 = random_2.normal(size=(dim_context_2, num_clusters_2))
        b_g_2 = random_2.normal(size=(1, num_clusters_2))
        phi_a_2 = random_2.choice(num_clusters_2, size=num_actions_item)
        theta_h_2 = random_2.normal(size=(dim_context_2, num_actions_item))
        M_h_2 = random_2.normal(size=(dim_context_2, num_actions_item))
        b_h_2 = random_2.normal(size=(1, num_actions_item))

        ## 評価方策の真の性能(policy value)を計算
        policy_value_2 = calc_true_value(
            dim_context=dim_context_2,
            num_actions=num_actions_item,
            num_clusters=num_clusters_2,
            lambda_=lambda_2,
            phi_a=phi_a_2,
            theta_g=theta_g_2,
            M_g=M_g_2,
            b_g=b_g_2,
            theta_h=theta_h_2,
            M_h=M_h_2,
            b_h=b_h_2,
        )

        estimated_policy_value_list_2 = []
        for _ in tqdm(range(num_runs_2), desc=f"num_actions={num_actions_item}..."):
            ## データ収集方策が形成する分布に従いログデータを生成
            offline_logged_data_2 = generate_synthetic_data(
                num_data=num_data_2,
                lambda_=lambda_2,
                beta=beta_2,
                theta_g=theta_g_2,
                M_g=M_g_2,
                b_g=b_g_2,
                theta_h=theta_h_2,
                M_h=M_h_2,
                b_h=b_h_2,
                phi_a=phi_a_2,
                dim_context=dim_context_2,
                num_actions=num_actions_item,
                num_clusters=num_clusters_2,
                random_state=_,
            )

            ## ログデータ上における評価方策の行動選択確率を計算
            pi_2 = eps_greedy_policy(offline_logged_data_2["q_x_a"])

            ## ログデータを用いてオフ方策評価を実行する
            estimated_policy_values_2 = dict()
            q_hat_2 = offline_logged_data_2["h_x_a"] + random_2.normal(size=(num_data_2, num_actions_item))
            estimated_policy_values_2["ips"] = calc_ips(offline_logged_data_2, pi_2)
            estimated_policy_values_2["mips"] = calc_mips(offline_logged_data_2, pi_2)
            estimated_policy_values_2["mips (selection)"] = calc_mips(
                offline_logged_data_2, pi_2, replace_c=np.int(num_clusters_2 / 2)
            )
            estimated_policy_values_2["offcem"] = calc_offcem(
                offline_logged_data_2, pi_2, q_hat_2, replace_c=np.int(num_clusters_2 / 2)
            )
            estimated_policy_value_list_2.append(estimated_policy_values_2)

        ## シミュレーション結果を集計する
        result_df_list_2.append(
            aggregate_simulation_results(
                estimated_policy_value_list_2,
                policy_value_2,
                "num_actions",
                num_actions_item,
            )
        )
    result_df_actions = pd.concat(result_df_list_2).reset_index(level=0)
    return (
        M_g_2,
        M_h_2,
        b_g_2,
        b_h_2,
        estimated_policy_value_list_2,
        estimated_policy_values_2,
        num_actions_item,
        num_clusters_2,
        offline_logged_data_2,
        phi_a_2,
        pi_2,
        policy_value_2,
        q_hat_2,
        result_df_actions,
        result_df_list_2,
        theta_g_2,
        theta_h_2,
    )


@app.cell
def __(mo):
    mo.md(r"""## 図3.16""")
    return


@app.cell
def __(num_actions_list, plt, remove_outliers, result_df_actions, sns, y_label_dict):
    fig3, ax_list3 = plt.subplots(1, 3, figsize=(35, 10), tight_layout=True)
    result_df_actions_ = remove_outliers(result_df_actions, ["ips", "mips", "offcem"])
    for i3, y3 in enumerate(["se", "bias", "variance"]):
        ax3 = ax_list3[i3]
        sns.lineplot(
            markers=True,
            markersize=40,
            linewidth=10,
            legend=False,
            style="est",
            x="num_actions",
            y=y3,
            hue="est",
            ax=ax3,
            ci=None,
            palette=["tab:red", "tab:orange", "tab:grey"],
            data=result_df_actions_.query("est != 'offcem'"),
        )
        ax3.set_title(y_label_dict[y3], fontsize=50)
        # yaxis
        ax3.set_ylabel("")
        ax3.set_ylim(0.0, 6)
        ax3.set_yticks([0, 2, 4, 6])
        ax3.tick_params(axis="y", labelsize=30)
        ax3.yaxis.set_label_coords(-0.1, 0.5)
        # xaxis
        ax3.set_xscale("log")
        if i3 == 1:
            ax3.set_xlabel(r"行動の数$|\mathcal{A}|$", fontsize=50)
        else:
            ax3.set_xlabel(r"", fontsize=40)
        ax3.set_xticks(num_actions_list)
        ax3.set_xticklabels(num_actions_list, fontsize=30)
        ax3.xaxis.set_label_coords(0.5, -0.12)
    fig3.legend(
        ["IPS", "MIPS (高次元行動特徴量をそのまま用いた場合)", "MIPS (行動の特徴量選択を行った場合)"],
        fontsize=50,
        bbox_to_anchor=(0.5, 1.15),
        ncol=4,
        loc="center",
    )
    plt.show()
    return ax3, ax_list3, fig3, i3, result_df_actions_, y3


@app.cell
def __(mo):
    mo.md(r"""## 図3.20""")
    return


@app.cell
def __(num_actions_list, plt, result_df_actions_, sns, y_label_dict):
    fig4, ax_list4 = plt.subplots(1, 3, figsize=(35, 10), tight_layout=True)
    for i4, y4 in enumerate(["se", "bias", "variance"]):
        ax4 = ax_list4[i4]
        sns.lineplot(
            markers=True,
            markersize=40,
            linewidth=10,
            legend=False,
            style="est",
            x="num_actions",
            y=y4,
            hue="est",
            ax=ax4,
            ci=None,
            palette=["tab:red", "tab:orange", "tab:purple"],
            data=result_df_actions_.query("est != 'mips (selection)'"),
        )
        ax4.set_title(y_label_dict[y4], fontsize=50)
        # yaxis
        ax4.set_ylabel("")
        ax4.set_ylim(0.0, 6)
        ax4.set_yticks([0, 2, 4, 6])
        ax4.tick_params(axis="y", labelsize=30)
        ax4.yaxis.set_label_coords(-0.1, 0.5)
        # xaxis
        ax4.set_xscale("log")
        if i4 == 1:
            ax4.set_xlabel(r"行動の数$|\mathcal{A}|$", fontsize=50)
        else:
            ax4.set_xlabel(r"", fontsize=40)
        ax4.set_xticks(num_actions_list)
        ax4.set_xticklabels(num_actions_list, fontsize=30)
        ax4.xaxis.set_label_coords(0.5, -0.12)
    fig4.legend(["IPS", "MIPS", "OffCEM"], fontsize=50, bbox_to_anchor=(0.5, 1.15), ncol=4, loc="center")
    plt.show()
    return ax4, ax_list4, fig4, i4, y4


@app.cell
def __(mo):
    mo.md(
        r"""### ログデータのサイズ$n$を変化させたときのIPS・MIPS・OffCEM推定量の平均二乗誤差・二乗バイアス・バリアンスの挙動"""
    )
    return


@app.cell
def __(check_random_state):
    ## シミュレーション設定
    num_runs_4 = 50  # シミュレーションの繰り返し回数
    dim_context_3 = 5  # 特徴量xの次元
    num_actions_3 = 500  # 行動数, |A|
    num_clusters_3 = 990  # 行動クラスタ数
    beta_3 = -0.2  # データ収集方策のパラメータ
    lambda_3 = 0.5  # クラスタ効果と残差効果の配合率
    random_state_3 = 12345
    random_3 = check_random_state(random_state_3)
    num_data_list = [250, 500, 1000, 2000, 4000]  # ログデータのサイズ
    return (
        beta_3,
        dim_context_3,
        lambda_3,
        num_actions_3,
        num_clusters_3,
        num_data_list,
        num_runs_3,
        random_3,
        random_state_3,
    )


@app.cell
def __(
    aggregate_simulation_results,
    beta_3,
    calc_ips,
    calc_mips,
    calc_offcem,
    calc_true_value,
    dim_context_3,
    eps_greedy_policy,
    generate_synthetic_data,
    lambda_3,
    num_actions_3,
    num_clusters_3,
    num_data_list,
    num_runs_3,
    pd,
    random_3,
    tqdm,
):
    result_df_list_3 = []
    ## 期待報酬関数を定義するためのパラメータを抽出
    phi_a_3 = random_3.choice(num_clusters_3, size=num_actions_3)
    theta_g_3 = random_3.normal(size=(dim_context_3, num_clusters_3))
    M_g_3 = random_3.normal(size=(dim_context_3, num_clusters_3))
    b_g_3 = random_3.normal(size=(1, num_clusters_3))
    theta_h_3 = random_3.normal(size=(dim_context_3, num_actions_3))
    M_h_3 = random_3.normal(size=(dim_context_3, num_actions_3))
    b_h_3 = random_3.normal(size=(1, num_actions_3))
    for num_data_item in num_data_list:
        ## 評価方策の真の性能(policy value)を計算
        policy_value_3 = calc_true_value(
            dim_context=dim_context_3,
            num_actions=num_actions_3,
            num_clusters=num_clusters_3,
            lambda_=lambda_3,
            phi_a=phi_a_3,
            theta_g=theta_g_3,
            M_g=M_g_3,
            b_g=b_g_3,
            theta_h=theta_h_3,
            M_h=M_h_3,
            b_h=b_h_3,
        )

        estimated_policy_value_list_3 = []
        for _ in tqdm(range(num_runs_3), desc=f"num_data={num_data_item}..."):
            ## データ収集方策が形成する分布に従いログデータを生成
            offline_logged_data_3 = generate_synthetic_data(
                num_data=num_data_item,
                lambda_=lambda_3,
                beta=beta_3,
                theta_g=theta_g_3,
                M_g=M_g_3,
                b_g=b_g_3,
                theta_h=theta_h_3,
                M_h=M_h_3,
                b_h=b_h_3,
                phi_a=phi_a_3,
                dim_context=dim_context_3,
                num_actions=num_actions_3,
                num_clusters=num_clusters_3,
                random_state=_,
            )

            ## ログデータ上における評価方策の行動選択確率を計算
            pi_3 = eps_greedy_policy(offline_logged_data_3["q_x_a"])

            ## ログデータを用いてオフ方策評価を実行する
            estimated_policy_values_3 = dict()
            q_hat_3 = offline_logged_data_3["h_x_a"] + random_3.normal(size=(num_data_item, num_actions_3))
            estimated_policy_values_3["ips"] = calc_ips(offline_logged_data_3, pi_3)
            estimated_policy_values_3["mips"] = calc_mips(offline_logged_data_3, pi_3)
            estimated_policy_values_3["mips (selection)"] = calc_mips(offline_logged_data_3, pi_3, replace_c=200)
            estimated_policy_values_3["offcem"] = calc_offcem(offline_logged_data_3, pi_3, q_hat_3, replace_c=200)
            estimated_policy_value_list_3.append(estimated_policy_values_3)

        ## シミュレーション結果を集計する
        result_df_list_3.append(
            aggregate_simulation_results(
                estimated_policy_value_list_3,
                policy_value_3,
                "num_data",
                num_data_item,
            )
        )
    result_df_data = pd.concat(result_df_list_3).reset_index(level=0)
    return (
        M_g_3,
        M_h_3,
        b_g_3,
        b_h_3,
        estimated_policy_value_list_3,
        estimated_policy_values_3,
        num_data_item,
        offline_logged_data_3,
        phi_a_3,
        pi_3,
        policy_value_3,
        q_hat_3,
        result_df_data,
        result_df_list_3,
        theta_g_3,
        theta_h_3,
    )


@app.cell
def __(mo):
    mo.md(r"""## 図3.17""")
    return


@app.cell
def __(num_data_list, plt, result_df_data, sns, y_label_dict):
    fig5, ax_list5 = plt.subplots(1, 3, figsize=(35, 10), tight_layout=True)
    for i5, y5 in enumerate(["se", "bias", "variance"]):
        ax5 = ax_list5[i5]
        sns.lineplot(
            markers=True,
            markersize=40,
            linewidth=10,
            legend=False,
            style="est",
            x="num_data",
            y=y5,
            hue="est",
            ax=ax5,
            ci=None,
            palette=["tab:red", "tab:orange", "tab:grey"],
            data=result_df_data.query("est != 'offcem'"),
        )
        ax5.set_title(y_label_dict[y5], fontsize=50)
        # yaxis
        ax5.set_ylabel("")
        ax5.set_ylim(0.0, 2.8)
        ax5.tick_params(axis="y", labelsize=30)
        ax5.yaxis.set_label_coords(-0.1, 0.5)
        # xaxis
        ax5.set_xscale("log")
        if i5 == 1:
            ax5.set_xlabel(r"ログデータのサイズ$n$", fontsize=50)
        else:
            ax5.set_xlabel(r"", fontsize=40)
        ax5.set_xticks(num_data_list)
        ax5.set_xticklabels(num_data_list, fontsize=30)
        ax5.xaxis.set_label_coords(0.5, -0.12)
    fig5.legend(
        ["IPS", "MIPS (高次元行動特徴量をそのまま用いた場合)", "MIPS (行動の特徴量選択を行った場合)"],
        fontsize=50,
        bbox_to_anchor=(0.5, 1.15),
        ncol=4,
        loc="center",
    )
    plt.show()
    return ax5, ax_list5, fig5, i5, y5


@app.cell
def __(mo):
    mo.md(r"""## 図3.21""")
    return


@app.cell
def __(num_data_list, plt, result_df_data, sns, y_label_dict):
    fig6, ax_list6 = plt.subplots(1, 3, figsize=(35, 10), tight_layout=True)
    for i6, y6 in enumerate(["se", "bias", "variance"]):
        ax6 = ax_list6[i6]
        sns.lineplot(
            markers=True,
            markersize=40,
            linewidth=10,
            legend=False,
            style="est",
            x="num_data",
            y=y6,
            hue="est",
            ax=ax6,
            ci=None,
            palette=["tab:red", "tab:orange", "tab:purple"],
            data=result_df_data.query("est != 'mips (selection)'"),
        )
        ax6.set_title(y_label_dict[y6], fontsize=50)
        # yaxis
        ax6.set_ylabel("")
        ax6.set_ylim(0.0, 2.8)
        ax6.tick_params(axis="y", labelsize=30)
        ax6.yaxis.set_label_coords(-0.1, 0.5)
        # xaxis
        ax6.set_xscale("log")
        if i6 == 1:
            ax6.set_xlabel(r"ログデータのサイズ$n$", fontsize=50)
        else:
            ax6.set_xlabel(r"", fontsize=40)
        ax6.set_xticks(num_data_list)
        ax6.set_xticklabels(num_data_list, fontsize=30)
        ax6.xaxis.set_label_coords(0.5, -0.12)
    fig6.legend(["IPS", "MIPS", "OffCEM"], fontsize=50, bbox_to_anchor=(0.5, 1.15), ncol=4, loc="center")
    plt.show()
    return ax6, ax_list6, fig6, i6, y6


@app.cell
def __():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
