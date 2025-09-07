import marimo

__generated_with = "0.9.24"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    mo.md(r"""# 2.3.2 Reward-interaction Inverse Propensity Score(RIPS)推定量
    参考文献
    - James McInerney, Brian Brost, Praveen Chandar, Rishabh Mehrotra, and Ben Carterette. [Counterfactual Evaluation of Slate Recommendations with Sequential Reward Interactions](https://arxiv.org/abs/2007.12986). KDD2020.
    - Haruka Kiyohara, Yuta Saito, Tatsuya Matsuhiro, Yusuke Narita, Nobuyuki Shimizu, and Yasuo Yamamoto. [Doubly Robust Off-Policy Evaluation for Ranking Policies under the Cascade Behavior Model](https://arxiv.org/abs/2202.01562). WSDM2022.""")
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
        r"""### カスケード性が成り立っている状況において、ランキングの長さ$K$を変化させたときのIPS・IIPS・RIPS推定量の平均二乗誤差・二乗バイアス・バリアンスの挙動"""
    )
    return


@app.cell
def __(check_random_state):
    ## シミュレーション設定
    num_runs_rips = 1000  # シミュレーションの繰り返し回数
    dim_context_rips = 5  # 特徴量xの次元
    num_data_rips = 2000  # ログデータのサイズ
    num_actions_rips = 3  # ユニークなアイテム数
    beta_rips = 1  # データ収集方策のパラメータ
    p_rips = [0, 1, 0]  # ユーザ行動の出現割合, (独立モデル, カスケードモデル, 全依存モデル)
    random_state_rips = 12345
    K_list_rips = [2, 4, 6, 8, 10, 12]  # ランキングの長さ
    return (
        K_list_rips,
        beta_rips,
        dim_context_rips,
        num_actions_rips,
        num_data_rips,
        num_runs_rips,
        p_rips,
        random_state_rips,
    )


@app.cell
def __(
    K_list_rips,
    aggregate_simulation_results,
    beta_rips,
    calc_avg,
    calc_iips,
    calc_ips,
    calc_rips,
    calc_true_value,
    check_random_state,
    dim_context_rips,
    eps_greedy_policy,
    generate_synthetic_data,
    num_actions_rips,
    num_data_rips,
    num_runs_rips,
    p_rips,
    pd,
    random_state_rips,
    tqdm,
):
    result_df_list_rips_k = []
    for K_rips in K_list_rips:
        ## 評価方策の真の性能(policy value)を計算
        random_rips = check_random_state(random_state_rips)
        theta_rips = random_rips.normal(size=(dim_context_rips, num_actions_rips))
        M_rips = random_rips.normal(size=(dim_context_rips, num_actions_rips))
        b_rips = random_rips.normal(size=(1, num_actions_rips))
        W_rips = random_rips.uniform(0, 1, size=(K_rips, K_rips))
        policy_value_rips_k = calc_true_value(
            dim_context=dim_context_rips,
            num_actions=num_actions_rips,
            theta=theta_rips,
            M=M_rips,
            b=b_rips,
            W=W_rips,
            K=K_rips,
            p=p_rips,
        )

        estimated_policy_value_list_rips_k = []
        for _ in tqdm(range(num_runs_rips), desc=f"K={K_rips}..."):
            ## データ収集方策が形成する分布に従いログデータを生成
            offline_logged_data_rips_k = generate_synthetic_data(
                num_data=num_data_rips,
                dim_context=dim_context_rips,
                num_actions=num_actions_rips,
                theta=theta_rips,
                M=M_rips,
                b=b_rips,
                W=W_rips,
                K=K_rips,
                p=p_rips,
                beta=beta_rips,
                random_state=_,
            )

            ## ログデータ上における評価方策の行動選択確率を計算
            pi_rips_k = eps_greedy_policy(offline_logged_data_rips_k["base_q_func"])

            ## ログデータを用いてオフ方策評価を実行する
            estimated_policy_values_rips_k = dict()
            estimated_policy_values_rips_k["avg"] = calc_avg(offline_logged_data_rips_k, pi_rips_k)
            estimated_policy_values_rips_k["ips"] = calc_ips(offline_logged_data_rips_k, pi_rips_k)
            estimated_policy_values_rips_k["iips"] = calc_iips(offline_logged_data_rips_k, pi_rips_k)
            estimated_policy_values_rips_k["rips"] = calc_rips(offline_logged_data_rips_k, pi_rips_k)
            estimated_policy_value_list_rips_k.append(estimated_policy_values_rips_k)

        ## シミュレーション結果を集計する
        result_df_list_rips_k.append(
            aggregate_simulation_results(
                estimated_policy_value_list_rips_k,
                policy_value_rips_k,
                "K",
                K_rips,
            )
        )
    result_df_rips_k = pd.concat(result_df_list_rips_k).reset_index(level=0)
    return (
        K_rips,
        M_rips,
        W_rips,
        b_rips,
        estimated_policy_value_list_rips_k,
        estimated_policy_values_rips_k,
        offline_logged_data_rips_k,
        pi_rips_k,
        policy_value_rips_k,
        random_rips,
        result_df_list_rips_k,
        result_df_rips_k,
        theta_rips,
    )


@app.cell
def __(mo):
    mo.md(r"""## 図2.19""")
    return


@app.cell
def __(K_list_rips, plt, result_df_rips_k, sns, y_label_dict):
    fig2_19, ax_list_19 = plt.subplots(1, 3, figsize=(35, 10), tight_layout=True)
    for i19, y19 in enumerate(["se", "bias", "variance"]):
        ax19 = ax_list_19[i19]
        sns.lineplot(
            markers=True,
            markersize=40,
            linewidth=7,
            legend=False,
            style="est",
            x="K",
            y=y19,
            hue="est",
            ax=ax19,
            ci=None,
            palette=["tab:red", "tab:blue", "tab:purple"],
            data=result_df_rips_k.query("est != 'avg'"),
        )
        ax19.set_title(y_label_dict[y19], fontsize=50)
        # yaxis
        ax19.set_ylabel("")
        ax19.set_ylim(0.0, 1)
        ax19.tick_params(axis="y", labelsize=30)
        ax19.yaxis.set_label_coords(-0.1, 0.5)
        # xaxis
        if i19 == 1:
            ax19.set_xlabel(r"ランキングの長さ$K$", fontsize=50)
        else:
            ax19.set_xlabel(r"", fontsize=40)
        ax19.set_xticks(K_list_rips)
        ax19.set_xticklabels(K_list_rips, fontsize=30)
        ax19.xaxis.set_label_coords(0.5, -0.12)
    fig2_19.legend(["IPS", "IIPS", "RIPS"], fontsize=50, bbox_to_anchor=(0.5, 1.15), ncol=5, loc="center")
    fig2_19
    return ax19, ax_list_19, fig2_19, i19, y19


@app.cell
def __(mo):
    mo.md(
        r"""### カスケード性が成り立っている状況において、ログデータのサイズ$n$を変化させたときのIPS・IIPS・RIPS推定量の平均二乗誤差・二乗バイアス・バリアンスの挙動"""
    )
    return


@app.cell
def __(check_random_state):
    ## シミュレーション設定
    num_runs_data_rips = 1000  # シミュレーションの繰り返し回数
    dim_context_data_rips = 5  # 特徴量xの次元
    K_data_rips = 5  # ランキングの長さ
    num_actions_data_rips = 4  # ユニークなアイテム数
    beta_data_rips = 1  # データ収集方策のパラメータ
    p_data_rips = [0, 1, 0]  # ユーザ行動の出現割合, (独立モデル, カスケードモデル, 全依存モデル)
    random_state_data_rips = 12345
    num_data_list_rips = [500, 1000, 2000, 4000, 8000]  # ログデータのサイズ
    return (
        K_data_rips,
        beta_data_rips,
        dim_context_data_rips,
        num_actions_data_rips,
        num_data_list_rips,
        num_runs_data_rips,
        p_data_rips,
        random_state_data_rips,
    )


@app.cell
def __(
    K_data_rips,
    aggregate_simulation_results,
    beta_data_rips,
    calc_avg,
    calc_iips,
    calc_ips,
    calc_rips,
    calc_true_value,
    check_random_state,
    dim_context_data_rips,
    eps_greedy_policy,
    generate_synthetic_data,
    num_actions_data_rips,
    num_data_list_rips,
    num_runs_data_rips,
    p_data_rips,
    pd,
    random_state_data_rips,
    tqdm,
):
    result_df_list_data_rips = []
    for num_data_rips in num_data_list_rips:
        ## 評価方策の真の性能(policy value)を計算
        random_data_rips = check_random_state(random_state_data_rips)
        theta_data_rips = random_data_rips.normal(size=(dim_context_data_rips, num_actions_data_rips))
        M_data_rips = random_data_rips.normal(size=(dim_context_data_rips, num_actions_data_rips))
        b_data_rips = random_data_rips.normal(size=(1, num_actions_data_rips))
        W_data_rips = random_data_rips.uniform(0, 1, size=(K_data_rips, K_data_rips))
        policy_value_data_rips = calc_true_value(
            dim_context=dim_context_data_rips,
            num_actions=num_actions_data_rips,
            theta=theta_data_rips,
            M=M_data_rips,
            b=b_data_rips,
            W=W_data_rips,
            K=K_data_rips,
            p=p_data_rips,
        )

        estimated_policy_value_list_data_rips = []
        for _ in tqdm(range(num_runs_data_rips), desc=f"num_data={num_data_rips}..."):
            ## データ収集方策が形成する分布に従いログデータを生成
            offline_logged_data_data_rips = generate_synthetic_data(
                num_data=num_data_rips,
                dim_context=dim_context_data_rips,
                num_actions=num_actions_data_rips,
                theta=theta_data_rips,
                M=M_data_rips,
                b=b_data_rips,
                W=W_data_rips,
                K=K_data_rips,
                p=p_data_rips,
                beta=beta_data_rips,
                random_state=_,
            )

            ## ログデータ上における評価方策の行動選択確率を計算
            pi_data_rips = eps_greedy_policy(offline_logged_data_data_rips["base_q_func"])

            ## ログデータを用いてオフ方策評価を実行する
            estimated_policy_values_data_rips = dict()
            estimated_policy_values_data_rips["avg"] = calc_avg(offline_logged_data_data_rips, pi_data_rips)
            estimated_policy_values_data_rips["ips"] = calc_ips(offline_logged_data_data_rips, pi_data_rips)
            estimated_policy_values_data_rips["iips"] = calc_iips(offline_logged_data_data_rips, pi_data_rips)
            estimated_policy_values_data_rips["rips"] = calc_rips(offline_logged_data_data_rips, pi_data_rips)
            estimated_policy_value_list_data_rips.append(estimated_policy_values_data_rips)

        ## シミュレーション結果を集計する
        result_df_list_data_rips.append(
            aggregate_simulation_results(
                estimated_policy_value_list_data_rips,
                policy_value_data_rips,
                "num_data",
                num_data_rips,
            )
        )
    result_df_data_rips = pd.concat(result_df_list_data_rips).reset_index(level=0)
    return (
        M_data_rips,
        W_data_rips,
        b_data_rips,
        estimated_policy_value_list_data_rips,
        estimated_policy_values_data_rips,
        num_data_rips,
        offline_logged_data_data_rips,
        pi_data_rips,
        policy_value_data_rips,
        random_data_rips,
        result_df_data_rips,
        result_df_list_data_rips,
        theta_data_rips,
    )


@app.cell
def __(mo):
    mo.md(r"""## 図2.20""")
    return


@app.cell
def __(num_data_list_rips, plt, result_df_data_rips, sns, y_label_dict):
    fig2_20, ax_list_20 = plt.subplots(1, 3, figsize=(35, 10), tight_layout=True)
    for i20, y20 in enumerate(["se", "bias", "variance"]):
        ax20 = ax_list_20[i20]
        sns.lineplot(
            markers=True,
            markersize=40,
            linewidth=7,
            legend=False,
            style="est",
            x="num_data",
            y=y20,
            hue="est",
            ax=ax20,
            ci=None,
            palette=["tab:red", "tab:blue", "tab:purple"],
            data=result_df_data_rips.query("est != 'avg'"),
        )
        ax20.set_title(y_label_dict[y20], fontsize=50)
        # yaxis
        ax20.set_ylabel("")
        ax20.set_ylim(0.0, 0.25)
        ax20.tick_params(axis="y", labelsize=30)
        ax20.yaxis.set_label_coords(-0.1, 0.5)
        # xaxis
        ax20.set_xscale("log")
        if i20 == 1:
            ax20.set_xlabel(r"ログデータのサイズ$n$", fontsize=50)
        else:
            ax20.set_xlabel(r"", fontsize=40)
        ax20.set_xticks(num_data_list_rips)
        ax20.set_xticklabels(num_data_list_rips, fontsize=30)
        ax20.xaxis.set_label_coords(0.5, -0.12)
    fig2_20.legend(["IPS", "IIPS", "RIPS"], fontsize=50, bbox_to_anchor=(0.5, 1.15), ncol=5, loc="center")
    fig2_20
    return ax20, ax_list_20, fig2_20, i20, y20


@app.cell
def __():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
