import marimo

__generated_with = "0.9.24"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    mo.md(r"""# 2.3.1 Independent Inverse Propensity Score(IIPS)推定量
    参考文献
    - Shuai Li, Yasin Abbasi-Yadkori, Branislav Kveton, S. Muthukrishnan, Vishwa Vinay, and Zheng Wen. [Offline Evaluation of Ranking Policies with Click Models](https://arxiv.org/abs/1804.10488). KDD2018.""")
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
    from estimators import calc_avg, calc_iips, calc_ips, calc_rips
    from utils import aggregate_simulation_results, eps_greedy_policy

    return (
        DataFrame,
        aggregate_simulation_results,
        calc_avg,
        calc_iips,
        calc_ips,
        calc_rips,
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
    mo.md(
        r"""### 独立性が成り立っている状況において、ランキングの長さ$K$を変化させたときのIPS・IIPS推定量の平均二乗誤差・二乗バイアス・バリアンスの挙動"""
    )
    return


@app.cell
def __(check_random_state):
    ## シミュレーション設定
    num_runs_iips = 1000  # シミュレーションの繰り返し回数
    dim_context_iips = 5  # 特徴量xの次元
    num_data_iips = 2000  # ログデータのサイズ
    num_actions_iips = 4  # ユニークなアイテム数
    beta_iips = 1  # データ収集方策のパラメータ
    p_iips = [1, 0, 0]  # ユーザ行動の出現割合, (独立モデル, カスケードモデル, 全依存モデル)
    random_state_iips = 12345
    K_list_iips = [2, 4, 6, 8, 10, 12]  # ランキングの長さ
    return (
        K_list_iips,
        beta_iips,
        dim_context_iips,
        num_actions_iips,
        num_data_iips,
        num_runs_iips,
        p_iips,
        random_state_iips,
    )


@app.cell
def __(
    K_list_iips,
    aggregate_simulation_results,
    beta_iips,
    calc_avg,
    calc_iips,
    calc_ips,
    calc_rips,
    calc_true_value,
    check_random_state,
    dim_context_iips,
    eps_greedy_policy,
    generate_synthetic_data,
    num_actions_iips,
    num_data_iips,
    num_runs_iips,
    p_iips,
    pd,
    random_state_iips,
    tqdm,
):
    result_df_list_iips_k = []
    for K_iips in K_list_iips:
        # 評価方策の真の性能(policy value)を計算
        random_iips = check_random_state(random_state_iips)
        theta_iips = random_iips.normal(size=(dim_context_iips, num_actions_iips))
        M_iips = random_iips.normal(size=(dim_context_iips, num_actions_iips))
        b_iips = random_iips.normal(size=(1, num_actions_iips))
        W_iips = random_iips.uniform(0, 1, size=(K_iips, K_iips))
        policy_value_iips_k = calc_true_value(
            dim_context=dim_context_iips,
            num_actions=num_actions_iips,
            theta=theta_iips,
            M=M_iips,
            b=b_iips,
            W=W_iips,
            K=K_iips,
            p=p_iips,
        )

        estimated_policy_value_list_iips_k = []
        for _ in tqdm(range(num_runs_iips), desc=f"K={K_iips}..."):
            ## データ収集方策が形成する分布に従いログデータを生成
            offline_logged_data_iips_k = generate_synthetic_data(
                num_data=num_data_iips,
                dim_context=dim_context_iips,
                num_actions=num_actions_iips,
                theta=theta_iips,
                M=M_iips,
                b=b_iips,
                W=W_iips,
                K=K_iips,
                p=p_iips,
                beta=beta_iips,
                random_state=_,
            )

            ## ログデータ上における評価方策の行動選択確率を計算
            pi_iips_k = eps_greedy_policy(offline_logged_data_iips_k["base_q_func"])

            ## ログデータを用いてオフ方策評価を実行する
            estimated_policy_values_iips_k = dict()
            estimated_policy_values_iips_k["avg"] = calc_avg(offline_logged_data_iips_k, pi_iips_k)
            estimated_policy_values_iips_k["ips"] = calc_ips(offline_logged_data_iips_k, pi_iips_k)
            estimated_policy_values_iips_k["iips"] = calc_iips(offline_logged_data_iips_k, pi_iips_k)
            estimated_policy_values_iips_k["rips"] = calc_rips(offline_logged_data_iips_k, pi_iips_k)
            estimated_policy_value_list_iips_k.append(estimated_policy_values_iips_k)

        ## シミュレーション結果を集計する
        result_df_list_iips_k.append(
            aggregate_simulation_results(
                estimated_policy_value_list_iips_k,
                policy_value_iips_k,
                "K",
                K_iips,
            )
        )
    result_df_iips_k = pd.concat(result_df_list_iips_k).reset_index(level=0)
    return (
        K_iips,
        M_iips,
        W_iips,
        b_iips,
        estimated_policy_value_list_iips_k,
        estimated_policy_values_iips_k,
        offline_logged_data_iips_k,
        pi_iips_k,
        policy_value_iips_k,
        random_iips,
        result_df_iips_k,
        result_df_list_iips_k,
        theta_iips,
    )


@app.cell
def __(mo):
    mo.md(r"""## 図2.12""")
    return


@app.cell
def __(K_list_iips, plt, result_df_iips_k, sns, y_label_dict):
    fig2_12, ax_list_12 = plt.subplots(1, 3, figsize=(35, 10), tight_layout=True)
    for i12, y12 in enumerate(["se", "bias", "variance"]):
        ax12 = ax_list_12[i12]
        sns.lineplot(
            markers=True,
            markersize=40,
            linewidth=7,
            legend=False,
            style="est",
            x="K",
            y=y12,
            hue="est",
            ax=ax12,
            ci=None,
            palette=["tab:red", "tab:blue"],
            data=result_df_iips_k.query("est != 'avg' and est != 'rips'"),
        )
        ax12.set_title(y_label_dict[y12], fontsize=50)
        # yaxis
        ax12.set_ylabel("")
        ax12.set_ylim(0.0, 0.17)
        ax12.tick_params(axis="y", labelsize=30)
        ax12.set_yticks([0.0, 0.04, 0.08, 0.12, 0.16])
        ax12.yaxis.set_label_coords(-0.1, 0.5)
        # xaxis
        if i12 == 1:
            ax12.set_xlabel(r"ランキングの長さ$K$", fontsize=50)
        else:
            ax12.set_xlabel(r"", fontsize=40)
        ax12.set_xticks(K_list_iips)
        ax12.set_xticklabels(K_list_iips, fontsize=30)
        ax12.xaxis.set_label_coords(0.5, -0.12)
    fig2_12.legend(["IPS", "IIPS"], fontsize=50, bbox_to_anchor=(0.5, 1.15), ncol=5, loc="center")
    fig2_12
    return ax12, ax_list_12, fig2_12, i12, y12


@app.cell
def __(mo):
    mo.md(
        r"""### 独立性が成り立っている状況において、ログデータのサイズ$n$を変化させたときのIPS・IIPS推定量の平均二乗誤差・二乗バイアス・バリアンスの挙動"""
    )
    return


@app.cell
def __(check_random_state):
    ## シミュレーション設定
    num_runs_data_iips = 1000  # シミュレーションの繰り返し回数
    dim_context_data_iips = 5  # 特徴量xの次元
    K_data_iips = 8  # ランキングの長さ
    num_actions_data_iips = 5  # ユニークなアイテム数
    beta_data_iips = 1  # データ収集方策のパラメータ
    p_data_iips = [1, 0, 0]  # ユーザ行動の出現割合, (独立モデル, カスケードモデル, 全依存モデル)
    random_state_data_iips = 12345
    num_data_list_iips = [500, 1000, 2000, 4000, 8000]  # ログデータのサイズ
    return (
        K_data_iips,
        beta_data_iips,
        dim_context_data_iips,
        num_actions_data_iips,
        num_data_list_iips,
        num_runs_data_iips,
        p_data_iips,
        random_state_data_iips,
    )


@app.cell
def __(
    K_data_iips,
    aggregate_simulation_results,
    beta_data_iips,
    calc_avg,
    calc_iips,
    calc_ips,
    calc_rips,
    calc_true_value,
    check_random_state,
    dim_context_data_iips,
    eps_greedy_policy,
    generate_synthetic_data,
    num_actions_data_iips,
    num_data_list_iips,
    num_runs_data_iips,
    p_data_iips,
    pd,
    random_state_data_iips,
    tqdm,
):
    result_df_list_data_iips = []
    for num_data_iips in num_data_list_iips:
        ## 評価方策の真の性能(policy value)を計算
        random_data_iips = check_random_state(random_state_data_iips)
        theta_data_iips = random_data_iips.normal(size=(dim_context_data_iips, num_actions_data_iips))
        M_data_iips = random_data_iips.normal(size=(dim_context_data_iips, num_actions_data_iips))
        b_data_iips = random_data_iips.normal(size=(1, num_actions_data_iips))
        W_data_iips = random_data_iips.uniform(0, 1, size=(K_data_iips, K_data_iips))
        policy_value_data_iips = calc_true_value(
            dim_context=dim_context_data_iips,
            num_actions=num_actions_data_iips,
            theta=theta_data_iips,
            M=M_data_iips,
            b=b_data_iips,
            W=W_data_iips,
            K=K_data_iips,
            p=p_data_iips,
        )

        estimated_policy_value_list_data_iips = []
        for _ in tqdm(range(num_runs_data_iips), desc=f"num_data={num_data_iips}..."):
            ## データ収集方策が形成する分布に従いログデータを生成
            offline_logged_data_data_iips = generate_synthetic_data(
                num_data=num_data_iips,
                dim_context=dim_context_data_iips,
                num_actions=num_actions_data_iips,
                theta=theta_data_iips,
                M=M_data_iips,
                b=b_data_iips,
                W=W_data_iips,
                K=K_data_iips,
                p=p_data_iips,
                beta=beta_data_iips,
                random_state=_,
            )

            ## ログデータ上における評価方策の行動選択確率を計算
            pi_data_iips = eps_greedy_policy(offline_logged_data_data_iips["base_q_func"])

            ## ログデータを用いてオフ方策評価を実行する
            estimated_policy_values_data_iips = dict()
            estimated_policy_values_data_iips["avg"] = calc_avg(offline_logged_data_data_iips, pi_data_iips)
            estimated_policy_values_data_iips["ips"] = calc_ips(offline_logged_data_data_iips, pi_data_iips)
            estimated_policy_values_data_iips["iips"] = calc_iips(offline_logged_data_data_iips, pi_data_iips)
            estimated_policy_values_data_iips["rips"] = calc_rips(offline_logged_data_data_iips, pi_data_iips)
            estimated_policy_value_list_data_iips.append(estimated_policy_values_data_iips)

        ## シミュレーション結果を集計する
        result_df_list_data_iips.append(
            aggregate_simulation_results(
                estimated_policy_value_list_data_iips,
                policy_value_data_iips,
                "num_data",
                num_data_iips,
            )
        )
    result_df_data_iips = pd.concat(result_df_list_data_iips).reset_index(level=0)
    return (
        M_data_iips,
        W_data_iips,
        b_data_iips,
        estimated_policy_value_list_data_iips,
        estimated_policy_values_data_iips,
        num_data_iips,
        offline_logged_data_data_iips,
        pi_data_iips,
        policy_value_data_iips,
        random_data_iips,
        result_df_data_iips,
        result_df_list_data_iips,
        theta_data_iips,
    )


@app.cell
def __(mo):
    mo.md(r"""## 図2.13""")
    return


@app.cell
def __(num_data_list_iips, plt, result_df_data_iips, sns, y_label_dict):
    fig2_13, ax_list_13 = plt.subplots(1, 3, figsize=(35, 10), tight_layout=True)
    for i13, y13 in enumerate(["se", "bias", "variance"]):
        ax13 = ax_list_13[i13]
        sns.lineplot(
            markers=True,
            markersize=40,
            linewidth=7,
            legend=False,
            style="est",
            x="num_data",
            y=y13,
            hue="est",
            ax=ax13,
            ci=None,
            palette=["tab:red", "tab:blue"],
            data=result_df_data_iips.query("est != 'avg' and est != 'rips'"),
        )
        ax13.set_title(y_label_dict[y13], fontsize=50)
        # yaxis
        ax13.set_ylabel("")
        ax13.set_ylim(0.0, 0.32)
        ax13.tick_params(axis="y", labelsize=30)
        ax13.set_yticks([0.0, 0.10, 0.20, 0.30])
        ax13.yaxis.set_label_coords(-0.1, 0.5)
        # xaxis
        ax13.set_xscale("log")
        if i13 == 1:
            ax13.set_xlabel(r"ログデータのサイズ$n$", fontsize=50)
        else:
            ax13.set_xlabel(r"", fontsize=40)
        ax13.set_xticks(num_data_list_iips)
        ax13.set_xticklabels(num_data_list_iips, fontsize=30)
        ax13.xaxis.set_label_coords(0.5, -0.12)
    fig2_13.legend(["IPS", "IIPS"], fontsize=50, bbox_to_anchor=(0.5, 1.15), ncol=5, loc="center")
    fig2_13
    return ax13, ax_list_13, fig2_13, i13, y13


@app.cell
def __():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
