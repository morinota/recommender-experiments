import marimo

__generated_with = "0.9.24"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    mo.md(r"""# 2.3.1-2.3.3 ユーザ行動モデルの出現割合を変化させたときの推定量の挙動
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
    from estimators import calc_aips, calc_avg, calc_iips, calc_ips, calc_rips
    from utils import aggregate_simulation_results, eps_greedy_policy

    return (
        DataFrame,
        aggregate_simulation_results,
        calc_aips,
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
        r"""### 独立モデルよりも複雑な行動モデルを持つユーザの割合を変化させたときのIPS・IIPS推定量の平均二乗誤差・二乗バイアス・バリアンスの挙動"""
    )
    return


@app.cell
def __(check_random_state):
    ## シミュレーション設定
    num_runs_behavior = 1000  # シミュレーションの繰り返し回数
    dim_context_behavior = 5  # 特徴量xの次元
    num_data_behavior = 3000  # ログデータのサイズ
    num_actions_behavior = 4  # ユニークなアイテム数
    K_behavior = 6  # ランキングの長さ
    beta_behavior = 1  # データ収集方策のパラメータ
    random_state_behavior = 12345
    p_complex_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # 独立性よりも複雑な行動モデルを持つユーザの割合
    return (
        K_behavior,
        beta_behavior,
        dim_context_behavior,
        num_actions_behavior,
        num_data_behavior,
        num_runs_behavior,
        p_complex_list,
        random_state_behavior,
    )


@app.cell
def __(
    K_behavior,
    aggregate_simulation_results,
    beta_behavior,
    calc_iips,
    calc_ips,
    calc_rips,
    calc_true_value,
    check_random_state,
    dim_context_behavior,
    eps_greedy_policy,
    generate_synthetic_data,
    num_actions_behavior,
    num_data_behavior,
    num_runs_behavior,
    p_complex_list,
    pd,
    random_state_behavior,
    tqdm,
):
    result_df_list_ind = []
    for p_complex in p_complex_list:
        ## 評価方策の真の性能(policy value)を計算
        random_ind = check_random_state(random_state_behavior)
        theta_ind = random_ind.normal(size=(dim_context_behavior, num_actions_behavior))
        M_ind = random_ind.normal(size=(dim_context_behavior, num_actions_behavior))
        b_ind = random_ind.normal(size=(1, num_actions_behavior))
        W_ind = random_ind.uniform(0, 1, size=(K_behavior, K_behavior))
        p_ind = [1 - p_complex, p_complex / 2, p_complex / 2]
        policy_value_ind = calc_true_value(
            dim_context=dim_context_behavior,
            num_actions=num_actions_behavior,
            theta=theta_ind,
            M=M_ind,
            b=b_ind,
            W=W_ind,
            K=K_behavior,
            p=p_ind,
        )

        estimated_policy_value_list_ind = []
        for _ in tqdm(range(num_runs_behavior), desc=f"p_complex={p_complex}..."):
            ## データ収集方策が形成する分布に従いログデータを生成
            offline_logged_data_ind = generate_synthetic_data(
                num_data=num_data_behavior,
                dim_context=dim_context_behavior,
                num_actions=num_actions_behavior,
                theta=theta_ind,
                M=M_ind,
                b=b_ind,
                W=W_ind,
                K=K_behavior,
                p=p_ind,
                beta=beta_behavior,
                random_state=_,
            )

            ## ログデータ上における評価方策の行動選択確率を計算
            pi_ind = eps_greedy_policy(offline_logged_data_ind["base_q_func"])

            ## ログデータを用いてオフ方策評価を実行する
            estimated_policy_values_ind = dict()
            estimated_policy_values_ind["ips"] = calc_ips(offline_logged_data_ind, pi_ind)
            estimated_policy_values_ind["iips"] = calc_iips(offline_logged_data_ind, pi_ind)
            estimated_policy_values_ind["rips"] = calc_rips(offline_logged_data_ind, pi_ind)
            estimated_policy_value_list_ind.append(estimated_policy_values_ind)

        ## シミュレーション結果を集計する
        result_df_list_ind.append(
            aggregate_simulation_results(
                estimated_policy_value_list_ind,
                policy_value_ind,
                "p_complex",
                p_complex,
            )
        )
    result_df_ind = pd.concat(result_df_list_ind).reset_index(level=0)
    return (
        M_ind,
        W_ind,
        b_ind,
        estimated_policy_value_list_ind,
        estimated_policy_values_ind,
        offline_logged_data_ind,
        p_complex,
        p_ind,
        pi_ind,
        policy_value_ind,
        random_ind,
        result_df_ind,
        result_df_list_ind,
        theta_ind,
    )


@app.cell
def __(mo):
    mo.md(r"""## 図2.14""")
    return


@app.cell
def __(p_complex_list, plt, result_df_ind, sns, y_label_dict):
    fig2_14, ax_list_14 = plt.subplots(1, 3, figsize=(35, 10), tight_layout=True)
    for i14, y14 in enumerate(["se", "bias", "variance"]):
        ax14 = ax_list_14[i14]
        sns.lineplot(
            markers=True,
            markersize=40,
            linewidth=7,
            legend=False,
            style="est",
            x="p_complex",
            y=y14,
            hue="est",
            ax=ax14,
            ci=None,
            palette=["tab:red", "tab:blue"],
            data=result_df_ind.query("est != 'avg' and est != 'aips' and est != 'rips'"),
        )
        ax14.set_title(y_label_dict[y14], fontsize=50)
        # yaxis
        ax14.set_ylabel("")
        ax14.set_ylim(0.0, 0.24)
        ax14.tick_params(axis="y", labelsize=30)
        ax14.set_yticks([0.0, 0.05, 0.10, 0.15, 0.20])
        ax14.yaxis.set_label_coords(-0.1, 0.5)
        # xaxis
        if y14 == "bias":
            ax14.set_xlabel(r"独立モデルよりも複雑な行動を持つユーザの割合", fontsize=50)
        else:
            ax14.set_xlabel("", fontsize=40)
        ax14.set_xticks(p_complex_list)
        ax14.set_xticklabels(p_complex_list, fontsize=30)
        ax14.xaxis.set_label_coords(0.5, -0.12)
    fig2_14.legend(["IPS", "IIPS"], fontsize=50, bbox_to_anchor=(0.5, 1.15), ncol=5, loc="center")
    fig2_14
    return ax14, ax_list_14, fig2_14, i14, y14


@app.cell
def __(mo):
    mo.md(
        r"""### カスケードモデルよりも複雑な行動モデルを持つユーザの割合を変化させたときのIPS・IIPS・RIPS推定量の平均二乗誤差・二乗バイアス・バリアンスの挙動"""
    )
    return


@app.cell
def __(check_random_state):
    ## シミュレーション設定
    num_runs_cascade = 1000  # シミュレーションの繰り返し回数
    dim_context_cascade = 5  # 特徴量xの次元
    num_data_cascade = 3000  # ログデータのサイズ
    num_actions_cascade = 4  # ユニークなアイテム数
    K_cascade = 6  # ランキングの長さ
    beta_cascade = 1  # データ収集方策のパラメータ
    random_state_cascade = 12345
    p_complex_list_cascade = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # カスケード性よりも複雑な行動モデルを持つユーザの割合
    return (
        K_cascade,
        beta_cascade,
        dim_context_cascade,
        num_actions_cascade,
        num_data_cascade,
        num_runs_cascade,
        p_complex_list_cascade,
        random_state_cascade,
    )


@app.cell
def __(
    K_cascade,
    aggregate_simulation_results,
    beta_cascade,
    calc_avg,
    calc_iips,
    calc_ips,
    calc_rips,
    calc_true_value,
    check_random_state,
    dim_context_cascade,
    eps_greedy_policy,
    generate_synthetic_data,
    num_actions_cascade,
    num_data_cascade,
    num_runs_cascade,
    p_complex_list_cascade,
    pd,
    random_state_cascade,
    tqdm,
):
    result_df_list_cascade = []
    for p_complex_cascade in p_complex_list_cascade:
        ## 評価方策の真の性能(policy value)を計算
        random_cascade = check_random_state(random_state_cascade)
        theta_cascade = random_cascade.normal(size=(dim_context_cascade, num_actions_cascade))
        M_cascade = random_cascade.normal(size=(dim_context_cascade, num_actions_cascade))
        b_cascade = random_cascade.normal(size=(1, num_actions_cascade))
        W_cascade = random_cascade.uniform(0, 1, size=(K_cascade, K_cascade))
        p_cascade = [0, 1 - p_complex_cascade, p_complex_cascade]
        policy_value_cascade = calc_true_value(
            dim_context=dim_context_cascade,
            num_actions=num_actions_cascade,
            theta=theta_cascade,
            M=M_cascade,
            b=b_cascade,
            W=W_cascade,
            K=K_cascade,
            p=p_cascade,
        )

        estimated_policy_value_list_cascade = []
        for _ in tqdm(range(num_runs_cascade), desc=f"p_complex={p_complex_cascade}..."):
            ## データ収集方策が形成する分布に従いログデータを生成
            offline_logged_data_cascade = generate_synthetic_data(
                num_data=num_data_cascade,
                dim_context=dim_context_cascade,
                num_actions=num_actions_cascade,
                theta=theta_cascade,
                M=M_cascade,
                b=b_cascade,
                W=W_cascade,
                K=K_cascade,
                p=p_cascade,
                beta=beta_cascade,
                random_state=_,
            )

            ## ログデータ上における評価方策の行動選択確率を計算
            pi_cascade = eps_greedy_policy(offline_logged_data_cascade["base_q_func"])

            ## ログデータを用いてオフ方策評価を実行する
            estimated_policy_values_cascade = dict()
            estimated_policy_values_cascade["avg"] = calc_avg(offline_logged_data_cascade, pi_cascade)
            estimated_policy_values_cascade["ips"] = calc_ips(offline_logged_data_cascade, pi_cascade)
            estimated_policy_values_cascade["iips"] = calc_iips(offline_logged_data_cascade, pi_cascade)
            estimated_policy_values_cascade["rips"] = calc_rips(offline_logged_data_cascade, pi_cascade)
            estimated_policy_value_list_cascade.append(estimated_policy_values_cascade)

        ## シミュレーション結果を集計する
        result_df_list_cascade.append(
            aggregate_simulation_results(
                estimated_policy_value_list_cascade,
                policy_value_cascade,
                "p_complex",
                p_complex_cascade,
            )
        )
    result_df_cascade = pd.concat(result_df_list_cascade).reset_index(level=0)
    return (
        M_cascade,
        W_cascade,
        b_cascade,
        estimated_policy_value_list_cascade,
        estimated_policy_values_cascade,
        offline_logged_data_cascade,
        p_cascade,
        p_complex_cascade,
        pi_cascade,
        policy_value_cascade,
        random_cascade,
        result_df_cascade,
        result_df_list_cascade,
        theta_cascade,
    )


@app.cell
def __(mo):
    mo.md(r"""## 図2.21""")
    return


@app.cell
def __(p_complex_list_cascade, plt, result_df_cascade, sns, y_label_dict):
    fig2_21, ax_list_21 = plt.subplots(1, 3, figsize=(35, 10), tight_layout=True)
    for i21, y21 in enumerate(["se", "bias", "variance"]):
        ax21 = ax_list_21[i21]
        sns.lineplot(
            markers=True,
            markersize=40,
            linewidth=7,
            legend=False,
            style="est",
            x="p_complex",
            y=y21,
            hue="est",
            ax=ax21,
            ci=None,
            palette=["tab:red", "tab:blue", "tab:purple"],
            data=result_df_cascade.query("est != 'avg' and est != 'aips'"),
        )
        ax21.set_title(y_label_dict[y21], fontsize=50)
        # yaxis
        ax21.set_ylabel("")
        ax21.set_ylim(0.0, 0.42)
        ax21.tick_params(axis="y", labelsize=30)
        ax21.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
        ax21.yaxis.set_label_coords(-0.1, 0.5)
        # xaxis
        if y21 == "bias":
            ax21.set_xlabel("カスケードモデルよりも複雑な行動を持つユーザの割合", fontsize=50)
        else:
            ax21.set_xlabel("", fontsize=40)
        ax21.set_xticks(p_complex_list_cascade)
        ax21.set_xticklabels(p_complex_list_cascade, fontsize=30)
        ax21.xaxis.set_label_coords(0.5, -0.12)
    fig2_21.legend(
        ["IPS", "IIPS", "RIPS"],
        fontsize=50,
        bbox_to_anchor=(0.5, 1.15),
        ncol=5,
        loc="center",
    )
    fig2_21
    return ax21, ax_list_21, fig2_21, i21, y21


@app.cell
def __(mo):
    mo.md(
        r"""### 複雑な行動モデルを持つユーザの割合を変化させたときのAIPS推定量の平均二乗誤差・二乗バイアス・バリアンスの挙動"""
    )
    return


@app.cell
def __(check_random_state):
    ## シミュレーション設定
    num_runs_aips_behavior = 1000  # シミュレーションの繰り返し回数
    dim_context_aips_behavior = 5  # 特徴量xの次元
    num_data_aips_behavior = 2000  # ログデータのサイズ
    num_actions_aips_behavior = 4  # ユニークなアイテム数
    K_aips_behavior = 8  # ランキングの長さ
    beta_aips_behavior = 1  # データ収集方策のパラメータ
    p_aips_behavior = [0.7, 0.2, 0.1]  # ユーザ行動の出現割合
    random_state_aips_behavior = 12345
    p_rand_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # 独立モデル, カスケードモデル以外のユーザ行動の出現割合
    max_k_dict = {0.0: 3, 0.2: 4, 0.4: 5, 0.6: 5, 0.8: 6, 1.0: 6}
    return (
        K_aips_behavior,
        beta_aips_behavior,
        dim_context_aips_behavior,
        max_k_dict,
        num_actions_aips_behavior,
        num_data_aips_behavior,
        num_runs_aips_behavior,
        p_aips_behavior,
        p_rand_list,
        random_state_aips_behavior,
    )


@app.cell
def __(
    K_aips_behavior,
    aggregate_simulation_results,
    beta_aips_behavior,
    calc_aips,
    calc_true_value,
    check_random_state,
    dim_context_aips_behavior,
    eps_greedy_policy,
    generate_synthetic_data,
    max_k_dict,
    num_actions_aips_behavior,
    num_data_aips_behavior,
    num_runs_aips_behavior,
    p_aips_behavior,
    p_rand_list,
    pd,
    random_state_aips_behavior,
    tqdm,
):
    result_df_list_rand = []
    for p_rand in p_rand_list:
        ## 評価方策の真の性能(policy value)を計算
        random_rand = check_random_state(random_state_aips_behavior)
        theta_rand = random_rand.normal(size=(dim_context_aips_behavior, num_actions_aips_behavior))
        M_rand = random_rand.normal(size=(dim_context_aips_behavior, num_actions_aips_behavior))
        b_rand = random_rand.normal(size=(1, num_actions_aips_behavior))
        W_rand = random_rand.uniform(0, 1, size=(K_aips_behavior, K_aips_behavior))
        policy_value_rand = calc_true_value(
            dim_context=dim_context_aips_behavior,
            num_actions=num_actions_aips_behavior,
            theta=theta_rand,
            M=M_rand,
            b=b_rand,
            W=W_rand,
            K=K_aips_behavior,
            p=p_aips_behavior,
            p_rand=p_rand,
        )

        estimated_policy_value_list_rand = []
        for _ in tqdm(range(num_runs_aips_behavior), desc=f"p_rand={p_rand}..."):
            ## データ収集方策が形成する分布に従いログデータを生成
            offline_logged_data_rand = generate_synthetic_data(
                num_data=num_data_aips_behavior,
                dim_context=dim_context_aips_behavior,
                num_actions=num_actions_aips_behavior,
                theta=theta_rand,
                M=M_rand,
                b=b_rand,
                W=W_rand,
                K=K_aips_behavior,
                p=p_aips_behavior,
                p_rand=p_rand,
                beta=beta_aips_behavior,
                random_state=_,
            )

            ## ログデータ上における評価方策の行動選択確率を計算
            pi_rand = eps_greedy_policy(offline_logged_data_rand["base_q_func"])

            ## ログデータを用いてオフ方策評価を実行する
            estimated_policy_values_rand = dict()
            estimated_policy_values_rand["aips"] = calc_aips(offline_logged_data_rand, pi_rand)
            estimated_policy_values_rand["aips (tune)"] = calc_aips(
                offline_logged_data_rand, pi_rand, max_k=max_k_dict[p_rand]
            )
            estimated_policy_value_list_rand.append(estimated_policy_values_rand)

        ## シミュレーション結果を集計する
        result_df_list_rand.append(
            aggregate_simulation_results(
                estimated_policy_value_list_rand,
                policy_value_rand,
                "p_rand",
                p_rand,
            )
        )
    result_df_rand = pd.concat(result_df_list_rand).reset_index(level=0)
    return (
        M_rand,
        W_rand,
        b_rand,
        estimated_policy_value_list_rand,
        estimated_policy_values_rand,
        offline_logged_data_rand,
        p_rand,
        pi_rand,
        policy_value_rand,
        random_rand,
        result_df_list_rand,
        result_df_rand,
        theta_rand,
    )


@app.cell
def __(mo):
    mo.md(r"""## 図2.26""")
    return


@app.cell
def __(p_rand_list, plt, result_df_rand, sns, y_label_dict):
    fig2_26, ax_list_26 = plt.subplots(1, 3, figsize=(35, 10), tight_layout=True)
    for i26, y26 in enumerate(["se", "bias", "variance"]):
        ax26 = ax_list_26[i26]
        sns.lineplot(
            markers=True,
            markersize=40,
            linewidth=9,
            legend=False,
            style="est",
            x="p_rand",
            y=y26,
            hue="est",
            ax=ax26,
            ci=None,
            palette=["tab:orange", "tab:pink"],
            data=result_df_rand.query("est == 'aips' or est == 'aips (tune)'"),
        )
        ax26.set_title(y_label_dict[y26], fontsize=50)
        # yaxis
        ax26.set_ylabel("")
        ax26.set_ylim(0.0, 0.30)
        ax26.set_yticks([0.0, 0.1, 0.2, 0.3])
        ax26.tick_params(axis="y", labelsize=30)
        ax26.yaxis.set_label_coords(-0.1, 0.5)
        # xaxis
        if i26 == 1:
            ax26.set_xlabel(r"複雑なユーザ行動の割合", fontsize=50)
        else:
            ax26.set_xlabel(r"", fontsize=40)
        ax26.set_xticks(p_rand_list)
        ax26.set_xticklabels(p_rand_list, fontsize=30)
        ax26.xaxis.set_label_coords(0.5, -0.12)
    fig2_26.legend(
        ["AIPS (真の行動モデルを用いた場合)", "AIPS (真ではない行動モデルをあえて用いた場合)"],
        fontsize=50,
        bbox_to_anchor=(0.5, 1.15),
        ncol=5,
        loc="center",
    )
    fig2_26
    return ax26, ax_list_26, fig2_26, i26, y26


@app.cell
def __():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
