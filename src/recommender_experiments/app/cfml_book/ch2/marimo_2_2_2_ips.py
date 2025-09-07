import marimo

__generated_with = "0.9.24"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    mo.md(r"""# 2.2.2  ランキングにおけるIPS推定量""")
    return


@app.cell
def __():
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
    from estimators import calc_avg, calc_ips
    from utils import aggregate_simulation_results, eps_greedy_policy

    return (
        DataFrame,
        aggregate_simulation_results,
        calc_avg,
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
        y_label_dict,
    )


@app.cell
def __(mo):
    mo.md(r"""### ランキングの長さ$K$を変化させたときのAVG・IPS推定量の平均二乗誤差・二乗バイアス・バリアンスの挙動""")
    return


@app.cell
def __(check_random_state):
    ## シミュレーション設定
    num_runs = 1000  # シミュレーションの繰り返し回数
    dim_context = 5  # 特徴量xの次元
    num_data_k = 2000  # ログデータのサイズ
    num_actions = 4  # ユニークなアイテム数
    beta = 1  # データ収集方策のパラメータ
    p = [0, 0, 1]  # ユーザ行動の出現割合, (独立モデル, カスケードモデル, 全依存モデル)
    random_state = 12345
    K_list = [4, 6, 8, 10, 12, 14]  # ランキングの長さ
    return (
        K_list,
        beta,
        dim_context,
        num_actions,
        num_data_k,
        num_runs,
        p,
        random_state,
    )


@app.cell
def __(
    K_list,
    aggregate_simulation_results,
    beta,
    calc_avg,
    calc_ips,
    calc_true_value,
    check_random_state,
    dim_context,
    eps_greedy_policy,
    generate_synthetic_data,
    num_actions,
    num_data_k,
    num_runs,
    p,
    pd,
    random_state,
    tqdm,
):
    result_df_list_k = []
    for K in K_list:
        ## 評価方策の真の性能(policy value)を計算
        random_ = check_random_state(random_state)
        theta = random_.normal(size=(dim_context, num_actions))
        M = random_.normal(size=(dim_context, num_actions))
        b = random_.normal(size=(1, num_actions))
        W = random_.uniform(0, 1, size=(K, K))
        policy_value_k = calc_true_value(
            dim_context=dim_context,
            num_actions=num_actions,
            theta=theta,
            M=M,
            b=b,
            W=W,
            K=K,
            p=p,
        )

        estimated_policy_value_list_k = []
        for _ in tqdm(range(num_runs), desc=f"K={K}..."):
            ## データ収集方策が形成する分布に従いログデータを生成
            offline_logged_data_k = generate_synthetic_data(
                num_data=num_data_k,
                dim_context=dim_context,
                num_actions=num_actions,
                theta=theta,
                M=M,
                b=b,
                W=W,
                K=K,
                p=p,
                beta=beta,
                random_state=_,
            )

            ## ログデータ上における評価方策の行動選択確率を計算
            pi_k = eps_greedy_policy(offline_logged_data_k["base_q_func"])

            ## ログデータを用いてオフ方策評価を実行する
            estimated_policy_values_k = dict()
            estimated_policy_values_k["avg"] = calc_avg(offline_logged_data_k, pi_k)
            estimated_policy_values_k["ips"] = calc_ips(offline_logged_data_k, pi_k)
            estimated_policy_value_list_k.append(estimated_policy_values_k)

        ## シミュレーション結果を集計する
        result_df_list_k.append(
            aggregate_simulation_results(
                estimated_policy_value_list_k,
                policy_value_k,
                "K",
                K,
            )
        )
    result_df_k = pd.concat(result_df_list_k).reset_index(level=0)
    return (
        K,
        M,
        W,
        b,
        estimated_policy_value_list_k,
        estimated_policy_values_k,
        offline_logged_data_k,
        pi_k,
        policy_value_k,
        random_,
        result_df_k,
        result_df_list_k,
        theta,
    )


@app.cell
def __(mo):
    mo.md(r"""## 図2.6""")
    return


@app.cell
def __(K_list, plt, result_df_k, sns, y_label_dict):
    fig2_6, ax_list_6 = plt.subplots(1, 3, figsize=(35, 10), tight_layout=True)
    for i6, y6 in enumerate(["se", "bias", "variance"]):
        ax6 = ax_list_6[i6]
        sns.lineplot(
            markers=True,
            markersize=40,
            linewidth=7,
            legend=False,
            style="est",
            x="K",
            y=y6,
            hue="est",
            ax=ax6,
            ci=None,
            palette=["tab:grey", "tab:red"],
            data=result_df_k.query("est == 'ips' or est == 'avg'"),
        )
        ax6.set_title(y_label_dict[y6], fontsize=50)
        # yaxis
        ax6.set_ylabel("")
        ax6.set_ylim(0.0, 65)
        ax6.tick_params(axis="y", labelsize=30)
        ax6.yaxis.set_label_coords(-0.1, 0.5)
        # xaxis
        if i6 == 1:
            ax6.set_xlabel(r"ランキングの長さ$K$", fontsize=50)
        else:
            ax6.set_xlabel(r"", fontsize=40)
        ax6.set_xticks(K_list)
        ax6.set_xticklabels(K_list, fontsize=30)
        ax6.xaxis.set_label_coords(0.5, -0.12)
    fig2_6.legend(["AVG", "IPS"], fontsize=50, bbox_to_anchor=(0.5, 1.15), ncol=5, loc="center")
    fig2_6
    return ax6, ax_list_6, fig2_6, i6, y6


@app.cell
def __(mo):
    mo.md(
        r"""### ログデータのサイズ$n$を変化させたときのAVG・IPS推定量の平均二乗誤差・二乗バイアス・バリアンスの挙動"""
    )
    return


@app.cell
def __(check_random_state):
    ## シミュレーション設定
    num_runs_data = 1000  # シミュレーションの繰り返し回数
    dim_context_data = 5  # 特徴量xの次元
    K_data = 8  # ランキングの長さ
    num_actions_data = 4  # ユニークなアイテム数
    beta_data = 1  # データ収集方策のパラメータ
    p_data = [0, 0, 1]  # ユーザ行動の出現割合, (独立モデル, カスケードモデル, 全依存モデル)
    random_state_data = 12345
    num_data_list = [500, 1000, 2000, 4000, 8000]  # ログデータのサイズ
    return (
        K_data,
        beta_data,
        dim_context_data,
        num_actions_data,
        num_data_list,
        num_runs_data,
        p_data,
        random_state_data,
    )


@app.cell
def __(
    K_data,
    aggregate_simulation_results,
    beta_data,
    calc_avg,
    calc_ips,
    calc_true_value,
    check_random_state,
    dim_context_data,
    eps_greedy_policy,
    generate_synthetic_data,
    num_actions_data,
    num_data_list,
    num_runs_data,
    p_data,
    pd,
    random_state_data,
    tqdm,
):
    result_df_list_data = []
    for num_data in num_data_list:
        ## 評価方策の真の性能(policy value)を計算
        random_data = check_random_state(random_state_data)
        theta_data = random_data.normal(size=(dim_context_data, num_actions_data))
        M_data = random_data.normal(size=(dim_context_data, num_actions_data))
        b_data = random_data.normal(size=(1, num_actions_data))
        W_data = random_data.uniform(0, 1, size=(K_data, K_data))
        policy_value_data = calc_true_value(
            dim_context=dim_context_data,
            num_actions=num_actions_data,
            theta=theta_data,
            M=M_data,
            b=b_data,
            W=W_data,
            K=K_data,
            p=p_data,
        )

        estimated_policy_value_list_data = []
        for _ in tqdm(range(num_runs_data), desc=f"num_data={num_data}..."):
            ## データ収集方策が形成する分布に従いログデータを生成
            offline_logged_data_data = generate_synthetic_data(
                num_data=num_data,
                dim_context=dim_context_data,
                num_actions=num_actions_data,
                theta=theta_data,
                M=M_data,
                b=b_data,
                W=W_data,
                K=K_data,
                p=p_data,
                beta=beta_data,
                random_state=_,
            )

            ## ログデータ上における評価方策の行動選択確率を計算
            pi_data = eps_greedy_policy(offline_logged_data_data["base_q_func"])

            ## ログデータを用いてオフ方策評価を実行する
            estimated_policy_values_data = dict()
            estimated_policy_values_data["avg"] = calc_avg(offline_logged_data_data, pi_data)
            estimated_policy_values_data["ips"] = calc_ips(offline_logged_data_data, pi_data)
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
        M_data,
        W_data,
        b_data,
        estimated_policy_value_list_data,
        estimated_policy_values_data,
        num_data,
        offline_logged_data_data,
        pi_data,
        policy_value_data,
        random_data,
        result_df_data,
        result_df_list_data,
        theta_data,
    )


@app.cell
def __(mo):
    mo.md(r"""## 図2.7""")
    return


@app.cell
def __(num_data_list, plt, result_df_data, sns, y_label_dict):
    fig2_7, ax_list_7 = plt.subplots(1, 3, figsize=(35, 10), tight_layout=True)
    for i7, y7 in enumerate(["se", "bias", "variance"]):
        ax7 = ax_list_7[i7]
        sns.lineplot(
            markers=True,
            markersize=40,
            linewidth=7,
            legend=False,
            style="est",
            x="num_data",
            y=y7,
            hue="est",
            ax=ax7,
            ci=None,
            palette=["tab:grey", "tab:red"],
            data=result_df_data.query("est == 'ips' or est == 'avg'"),
        )
        ax7.set_title(y_label_dict[y7], fontsize=50)
        # yaxis
        ax7.set_ylabel("")
        ax7.set_ylim(0.0, 6.5)
        ax7.tick_params(axis="y", labelsize=30)
        ax7.yaxis.set_label_coords(-0.1, 0.5)
        # xaxis
        ax7.set_xscale("log")
        if i7 == 1:
            ax7.set_xlabel(r"ログデータのサイズ$n$", fontsize=50)
        else:
            ax7.set_xlabel(r"", fontsize=40)
        ax7.set_xticks(num_data_list)
        ax7.set_xticklabels(num_data_list, fontsize=30)
        ax7.xaxis.set_label_coords(0.5, -0.12)
    fig2_7.legend(["AVG", "IPS"], fontsize=50, bbox_to_anchor=(0.5, 1.15), ncol=5, loc="center")
    fig2_7
    return ax7, ax_list_7, fig2_7, i7, y7


@app.cell
def __():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
