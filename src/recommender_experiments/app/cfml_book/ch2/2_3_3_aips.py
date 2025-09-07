import marimo

__generated_with = "0.9.24"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    mo.md(r"""# 2.3.3 Adaptive Inverse Propensity Score(AIPS)推定量
    参考文献
    - Haruka Kiyohara, Masatoshi Uehara, Yusuke Narita, Nobuyuki Shimizu, Yasuo Yamamoto, and Yuta Saito. [Off-Policy Evaluation of Ranking Policies under Diverse User Behavior](https://arxiv.org/abs/2306.15098). KDD2023.""")
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
    from estimators import calc_aips, calc_iips, calc_ips, calc_rips
    from utils import aggregate_simulation_results, eps_greedy_policy

    return (
        DataFrame,
        aggregate_simulation_results,
        calc_aips,
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
        r"""### あらゆる行動モデルが混在する状況において、ランキングの長さ$K$を変化させたときのIPS・RIPS・AIPS推定量の平均二乗誤差・二乗バイアス・バリアンスの挙動"""
    )
    return


@app.cell
def __(check_random_state):
    ## シミュレーション設定
    num_runs_aips = 1000  # シミュレーションの繰り返し回数
    dim_context_aips = 5  # 特徴量xの次元
    num_data_aips = 2000  # ログデータのサイズ
    num_actions_aips = 3  # ユニークなアイテム数
    beta_aips = 1  # データ収集方策のパラメータ
    p_aips = [0.7, 0.2, 0.1]  # ユーザ行動の出現割合, (独立モデル, カスケードモデル, 全依存モデル)
    p_rand_aips = 0.8  # 独立モデル, カスケードモデル以外のユーザ行動の出現割合
    random_state_aips = 12345
    K_list_aips = [6, 8, 10, 12, 14, 16]  # ランキングの長さ
    return (
        K_list_aips,
        beta_aips,
        dim_context_aips,
        num_actions_aips,
        num_data_aips,
        num_runs_aips,
        p_aips,
        p_rand_aips,
        random_state_aips,
    )


@app.cell
def __(
    K_list_aips,
    aggregate_simulation_results,
    beta_aips,
    calc_aips,
    calc_iips,
    calc_ips,
    calc_rips,
    calc_true_value,
    check_random_state,
    dim_context_aips,
    eps_greedy_policy,
    generate_synthetic_data,
    num_actions_aips,
    num_data_aips,
    num_runs_aips,
    p_aips,
    p_rand_aips,
    pd,
    random_state_aips,
    tqdm,
):
    result_df_list_aips_k = []
    for K_aips in K_list_aips:
        ## 評価方策の真の性能(policy value)を計算
        random_aips = check_random_state(random_state_aips)
        theta_aips = random_aips.normal(size=(dim_context_aips, num_actions_aips))
        M_aips = random_aips.normal(size=(dim_context_aips, num_actions_aips))
        b_aips = random_aips.normal(size=(1, num_actions_aips))
        W_aips = random_aips.uniform(0, 1, size=(K_aips, K_aips))
        policy_value_aips_k = calc_true_value(
            dim_context=dim_context_aips,
            num_actions=num_actions_aips,
            theta=theta_aips,
            M=M_aips,
            b=b_aips,
            W=W_aips,
            K=K_aips,
            p=p_aips,
            p_rand=p_rand_aips,
        )

        estimated_policy_value_list_aips_k = []
        for _ in tqdm(range(num_runs_aips), desc=f"K={K_aips}..."):
            ## データ収集方策が形成する分布に従いログデータを生成
            offline_logged_data_aips_k = generate_synthetic_data(
                num_data=num_data_aips,
                dim_context=dim_context_aips,
                num_actions=num_actions_aips,
                theta=theta_aips,
                M=M_aips,
                b=b_aips,
                W=W_aips,
                K=K_aips,
                p=p_aips,
                p_rand=p_rand_aips,
                beta=beta_aips,
                random_state=_,
            )

            ## ログデータ上における評価方策の行動選択確率を計算
            pi_aips_k = eps_greedy_policy(offline_logged_data_aips_k["base_q_func"])

            ## ログデータを用いてオフ方策評価を実行する
            estimated_policy_values_aips_k = dict()
            estimated_policy_values_aips_k["ips"] = calc_ips(offline_logged_data_aips_k, pi_aips_k)
            estimated_policy_values_aips_k["iips"] = calc_iips(offline_logged_data_aips_k, pi_aips_k)
            estimated_policy_values_aips_k["rips"] = calc_rips(offline_logged_data_aips_k, pi_aips_k)
            estimated_policy_values_aips_k["aips"] = calc_aips(offline_logged_data_aips_k, pi_aips_k)
            estimated_policy_value_list_aips_k.append(estimated_policy_values_aips_k)

        ## シミュレーション結果を集計する
        result_df_list_aips_k.append(
            aggregate_simulation_results(
                estimated_policy_value_list_aips_k,
                policy_value_aips_k,
                "K",
                K_aips,
            )
        )
    result_df_aips_k = pd.concat(result_df_list_aips_k).reset_index(level=0)
    return (
        K_aips,
        M_aips,
        W_aips,
        b_aips,
        estimated_policy_value_list_aips_k,
        estimated_policy_values_aips_k,
        offline_logged_data_aips_k,
        pi_aips_k,
        policy_value_aips_k,
        random_aips,
        result_df_aips_k,
        result_df_list_aips_k,
        theta_aips,
    )


@app.cell
def __(mo):
    mo.md(r"""## 図2.24""")
    return


@app.cell
def __(K_list_aips, plt, result_df_aips_k, sns, y_label_dict):
    fig2_24, ax_list_24 = plt.subplots(1, 3, figsize=(35, 10), tight_layout=True)
    for i24, y24 in enumerate(["se", "bias", "variance"]):
        ax24 = ax_list_24[i24]
        sns.lineplot(
            markers=True,
            markersize=40,
            linewidth=7,
            legend=False,
            style="est",
            x="K",
            y=y24,
            hue="est",
            ax=ax24,
            ci=None,
            palette=["tab:red", "tab:purple", "tab:orange"],
            data=result_df_aips_k.query("est != 'aips (tune)' and est != 'iips'"),
        )
        ax24.set_title(y_label_dict[y24], fontsize=50)
        # yaxis
        ax24.set_ylabel("")
        ax24.set_ylim(0.0, 3.2)
        ax24.tick_params(axis="y", labelsize=30)
        ax24.set_yticks([0.0, 0.5, 1, 1.5, 2.0, 2.5, 3.0])
        ax24.yaxis.set_label_coords(-0.1, 0.5)
        # xaxis
        if i24 == 1:
            ax24.set_xlabel(r"ランキングの長さ$K$", fontsize=50)
        else:
            ax24.set_xlabel(r"", fontsize=40)
        ax24.set_xticks(K_list_aips)
        ax24.set_xticklabels(K_list_aips, fontsize=30)
        ax24.xaxis.set_label_coords(0.5, -0.12)
    fig2_24.legend(["IPS", "RIPS", "AIPS"], fontsize=50, bbox_to_anchor=(0.5, 1.15), ncol=5, loc="center")
    fig2_24
    return ax24, ax_list_24, fig2_24, i24, y24


@app.cell
def __(mo):
    mo.md(
        r"""### あらゆる行動モデルが混在する状況において、ログデータのサイズ$n$を変化させたときのIPS・RIPS・AIPS推定量の平均二乗誤差・二乗バイアス・バリアンスの挙動"""
    )
    return


@app.cell
def __(check_random_state, p_rand_aips):
    ## シミュレーション設定
    num_runs_data_aips = 1000  # シミュレーションの繰り返し回数
    dim_context_data_aips = 5  # 特徴量xの次元
    K_data_aips = 8  # ランキングの長さ
    num_actions_data_aips = 3  # ユニークなアイテム数
    beta_data_aips = 1  # データ収集方策のパラメータ
    p_data_aips = [0.7, 0.2, 0.1]  # ユーザ行動の出現割合, (独立モデル, カスケードモデル, 全依存モデル)
    random_state_data_aips = 12345
    num_data_list_aips = [500, 1000, 2000, 4000, 8000]  # ログデータのサイズ
    return (
        K_data_aips,
        beta_data_aips,
        dim_context_data_aips,
        num_actions_data_aips,
        num_data_list_aips,
        num_runs_data_aips,
        p_data_aips,
        random_state_data_aips,
    )


@app.cell
def __(
    K_data_aips,
    aggregate_simulation_results,
    beta_data_aips,
    calc_aips,
    calc_iips,
    calc_ips,
    calc_rips,
    calc_true_value,
    check_random_state,
    dim_context_data_aips,
    eps_greedy_policy,
    generate_synthetic_data,
    num_actions_data_aips,
    num_data_list_aips,
    num_runs_data_aips,
    p_data_aips,
    p_rand_aips,
    pd,
    random_state_data_aips,
    tqdm,
):
    result_df_list_data_aips = []
    for num_data_aips in num_data_list_aips:
        ## 評価方策の真の性能(policy value)を計算
        random_data_aips = check_random_state(random_state_data_aips)
        theta_data_aips = random_data_aips.normal(size=(dim_context_data_aips, num_actions_data_aips))
        M_data_aips = random_data_aips.normal(size=(dim_context_data_aips, num_actions_data_aips))
        b_data_aips = random_data_aips.normal(size=(1, num_actions_data_aips))
        W_data_aips = random_data_aips.uniform(0, 1, size=(K_data_aips, K_data_aips))
        policy_value_data_aips = calc_true_value(
            dim_context=dim_context_data_aips,
            num_actions=num_actions_data_aips,
            theta=theta_data_aips,
            M=M_data_aips,
            b=b_data_aips,
            W=W_data_aips,
            K=K_data_aips,
            p=p_data_aips,
            p_rand=p_rand_aips,
        )

        estimated_policy_value_list_data_aips = []
        for _ in tqdm(range(num_runs_data_aips), desc=f"num_data={num_data_aips}..."):
            ## データ収集方策が形成する分布に従いログデータを生成
            offline_logged_data_data_aips = generate_synthetic_data(
                num_data=num_data_aips,
                dim_context=dim_context_data_aips,
                num_actions=num_actions_data_aips,
                theta=theta_data_aips,
                M=M_data_aips,
                b=b_data_aips,
                W=W_data_aips,
                K=K_data_aips,
                p=p_data_aips,
                p_rand=p_rand_aips,
                beta=beta_data_aips,
                random_state=_,
            )

            ## ログデータ上における評価方策の行動選択確率を計算
            pi_data_aips = eps_greedy_policy(offline_logged_data_data_aips["base_q_func"])

            ## ログデータを用いてオフ方策評価を実行する
            estimated_policy_values_data_aips = dict()
            estimated_policy_values_data_aips["ips"] = calc_ips(offline_logged_data_data_aips, pi_data_aips)
            estimated_policy_values_data_aips["iips"] = calc_iips(offline_logged_data_data_aips, pi_data_aips)
            estimated_policy_values_data_aips["rips"] = calc_rips(offline_logged_data_data_aips, pi_data_aips)
            estimated_policy_values_data_aips["aips"] = calc_aips(offline_logged_data_data_aips, pi_data_aips)
            estimated_policy_value_list_data_aips.append(estimated_policy_values_data_aips)

        ## シミュレーション結果を集計する
        result_df_list_data_aips.append(
            aggregate_simulation_results(
                estimated_policy_value_list_data_aips,
                policy_value_data_aips,
                "num_data",
                num_data_aips,
            )
        )
    result_df_data_aips = pd.concat(result_df_list_data_aips).reset_index(level=0)
    return (
        M_data_aips,
        W_data_aips,
        b_data_aips,
        estimated_policy_value_list_data_aips,
        estimated_policy_values_data_aips,
        num_data_aips,
        offline_logged_data_data_aips,
        pi_data_aips,
        policy_value_data_aips,
        random_data_aips,
        result_df_data_aips,
        result_df_list_data_aips,
        theta_data_aips,
    )


@app.cell
def __(mo):
    mo.md(r"""## 図2.25""")
    return


@app.cell
def __(num_data_list_aips, plt, result_df_data_aips, sns, y_label_dict):
    fig2_25, ax_list_25 = plt.subplots(1, 3, figsize=(35, 10), tight_layout=True)
    for i25, y25 in enumerate(["se", "bias", "variance"]):
        ax25 = ax_list_25[i25]
        sns.lineplot(
            markers=True,
            markersize=40,
            linewidth=7,
            legend=False,
            style="est",
            x="num_data",
            y=y25,
            hue="est",
            ax=ax25,
            ci=None,
            palette=["tab:red", "tab:purple", "tab:orange"],
            data=result_df_data_aips.query("est != 'aips (tune)' and est != 'iips'"),
        )
        ax25.set_title(y_label_dict[y25], fontsize=50)
        # yaxis
        ax25.set_ylabel("")
        ax25.set_ylim(0.0, 0.4)
        ax25.tick_params(axis="y", labelsize=30)
        ax25.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
        ax25.yaxis.set_label_coords(-0.1, 0.5)
        # xaxis
        ax25.set_xscale("log")
        if i25 == 1:
            ax25.set_xlabel(r"ログデータのサイズ$n$", fontsize=50)
        else:
            ax25.set_xlabel(r"", fontsize=40)
        ax25.set_xticks(num_data_list_aips)
        ax25.set_xticklabels(num_data_list_aips, fontsize=30)
        ax25.xaxis.set_label_coords(0.5, -0.12)
    fig2_25.legend(["IPS", "RIPS", "AIPS"], fontsize=50, bbox_to_anchor=(0.5, 1.15), ncol=5, loc="center")
    fig2_25
    return ax25, ax_list_25, fig2_25, i25, y25


@app.cell
def __():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
