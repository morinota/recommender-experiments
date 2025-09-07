import marimo

__generated_with = "0.9.24"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    mo.md(r"""## 2.3.1-2.3.2 ランキングにおける各種重要度重みの大きさを比較""")
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

    from dataset import calc_true_value, generate_synthetic_data
    from estimators import calc_weights
    from utils import eps_greedy_policy

    return (
        DataFrame,
        calc_true_value,
        calc_weights,
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
    )


@app.cell
def __(check_random_state):
    ## シミュレーション設定
    num_runs_compare = 1000  # シミュレーションの繰り返し回数
    dim_context_compare = 5  # 特徴量xの次元
    num_data_compare = 2000  # ログデータのサイズ
    beta_compare = 1  # データ収集方策のパラメータ
    p_compare = [0, 0, 1]  # ユーザ行動の出現割合
    random_state_compare = 12345
    num_actions_list = [2, 4, 6, 8, 10, 12]  # ユニークなアイテム数, |A|
    K_list_compare = [2, 4, 6, 8, 10, 12]  # ランキングの長さ
    return (
        K_list_compare,
        beta_compare,
        dim_context_compare,
        num_actions_list,
        num_data_compare,
        num_runs_compare,
        p_compare,
        random_state_compare,
    )


@app.cell
def __(
    DataFrame,
    K_list_compare,
    beta_compare,
    calc_weights,
    check_random_state,
    dim_context_compare,
    eps_greedy_policy,
    generate_synthetic_data,
    num_actions_list,
    num_data_compare,
    num_runs_compare,
    p_compare,
    pd,
    random_state_compare,
    tqdm,
):
    result_df_list_actions = []
    K_actions = 8  # ランキングの長さ (デフォルト)
    for num_actions in num_actions_list:
        random_actions = check_random_state(random_state_compare)
        theta_actions = random_actions.normal(size=(dim_context_compare, num_actions))
        M_actions = random_actions.normal(size=(dim_context_compare, num_actions))
        b_actions = random_actions.normal(size=(1, num_actions))
        W_actions = random_actions.uniform(0, 1, size=(K_actions, K_actions))

        for _ in tqdm(range(num_runs_compare), desc=f"num_actions={num_actions}..."):
            ## データ収集方策が形成する分布に従いログデータを生成
            offline_logged_data_actions = generate_synthetic_data(
                num_data=num_data_compare,
                dim_context=dim_context_compare,
                num_actions=num_actions,
                theta=theta_actions,
                M=M_actions,
                b=b_actions,
                W=W_actions,
                K=K_actions,
                p=p_compare,
                beta=beta_compare,
                random_state=_,
            )

            ## ログデータ上における評価方策の行動選択確率を計算
            pi_actions = eps_greedy_policy(offline_logged_data_actions["base_q_func"])

            ## シミュレーション結果を集計
            rank_weight_max, pos_weight_max, topk_weight_max = calc_weights(offline_logged_data_actions, pi_actions)
            result_df_actions = DataFrame(
                [num_actions, rank_weight_max, pos_weight_max, topk_weight_max],
                index=["num_actions", "max_rank_weight", "max_pos_weight", "max_topk_weight"],
            )
            result_df_list_actions.append(result_df_actions.T)
    result_df_actions_final = pd.concat(result_df_list_actions).reset_index(level=0).groupby(["num_actions"]).mean()
    return (
        K_actions,
        M_actions,
        W_actions,
        b_actions,
        num_actions,
        offline_logged_data_actions,
        pi_actions,
        pos_weight_max,
        random_actions,
        rank_weight_max,
        result_df_actions,
        result_df_actions_final,
        result_df_list_actions,
        theta_actions,
        topk_weight_max,
    )


@app.cell
def __(
    DataFrame,
    K_list_compare,
    beta_compare,
    calc_weights,
    check_random_state,
    dim_context_compare,
    eps_greedy_policy,
    generate_synthetic_data,
    num_data_compare,
    num_runs_compare,
    p_compare,
    pd,
    random_state_compare,
    tqdm,
):
    result_df_list_k = []
    num_actions_k = 5  # ユニークなアイテム数 (デフォルト)
    for K_compare in K_list_compare:
        random_k = check_random_state(random_state_compare)
        theta_k = random_k.normal(size=(dim_context_compare, num_actions_k))
        M_k = random_k.normal(size=(dim_context_compare, num_actions_k))
        b_k = random_k.normal(size=(1, num_actions_k))
        W_k = random_k.uniform(0, 1, size=(K_compare, K_compare))

        for _ in tqdm(range(num_runs_compare), desc=f"K={K_compare}..."):
            ## データ収集方策が形成する分布に従いログデータを生成
            offline_logged_data_k = generate_synthetic_data(
                num_data=num_data_compare,
                dim_context=dim_context_compare,
                num_actions=num_actions_k,
                theta=theta_k,
                M=M_k,
                b=b_k,
                W=W_k,
                K=K_compare,
                p=p_compare,
                beta=beta_compare,
                random_state=_,
            )

            ## ログデータ上における評価方策の行動選択確率を計算
            pi_k = eps_greedy_policy(offline_logged_data_k["base_q_func"])

            ## シミュレーション結果を集計
            rank_weight_max_k, pos_weight_max_k, topk_weight_max_k = calc_weights(offline_logged_data_k, pi_k)
            result_df_k_item = DataFrame(
                [K_compare, rank_weight_max_k, pos_weight_max_k, topk_weight_max_k],
                index=["K", "max_rank_weight", "max_pos_weight", "max_topk_weight"],
            )
            result_df_list_k.append(result_df_k_item.T)
    result_df_k_final = pd.concat(result_df_list_k).reset_index(level=0).groupby(["K"]).mean()
    return (
        K_compare,
        M_k,
        W_k,
        b_k,
        num_actions_k,
        offline_logged_data_k,
        pi_k,
        pos_weight_max_k,
        random_k,
        rank_weight_max_k,
        result_df_k_final,
        result_df_k_item,
        result_df_list_k,
        theta_k,
        topk_weight_max_k,
    )


@app.cell
def __(mo):
    mo.md(r"""## 図2.11""")
    return


@app.cell
def __(K_list_compare, num_actions_list, plt, result_df_actions_final, result_df_k_final, sns):
    fig2_11, ax_list_11 = plt.subplots(1, 2, figsize=(18, 7), tight_layout=True)
    for i11, df11 in enumerate([result_df_actions_final, result_df_k_final]):
        ax11 = ax_list_11[i11]
        x11 = "num_actions" if i11 == 0 else "K"
        for y11 in ["max_rank_weight", "max_pos_weight"]:
            if y11 == "max_pos_weight":
                marker11 = "X"
                line11 = "--"
            elif y11 == "max_topk_weight":
                marker11 = "*"
                line11 = "---"
            else:
                marker11 = "o"
                line11 = "-"
            sns.lineplot(
                markers=True,
                markersize=25,
                linewidth=7,
                legend=False,
                marker=marker11,
                linestyle=line11,
                x=x11,
                y=y11,
                ax=ax11,
                data=df11,
            )
        # yaxis
        ax11.set_ylabel("")
        ax11.set_ylim(0.0, 585)
        ax11.tick_params(axis="y", labelsize=20)
        ax11.yaxis.set_label_coords(-0.1, 0.5)
        # xaxis
        if i11 == 0:
            ax11.set_xlabel(r"ユニークな行動の数$|\mathcal{A}|$", fontsize=28)
            ax11.set_xticks(num_actions_list)
            ax11.set_xticklabels(num_actions_list, fontsize=22)
        else:
            ax11.set_xlabel(r"ランキングの長さ$K$", fontsize=28)
            ax11.set_xticks(K_list_compare)
            ax11.set_xticklabels(K_list_compare, fontsize=22)
        ax11.xaxis.set_label_coords(0.5, -0.12)
    fig2_11.legend(
        ["ランキングレベルの重要度重み", "ポジションレベルの重要度重み"],
        fontsize=28,
        bbox_to_anchor=(0.5, 1.05),
        ncol=2,
        loc="center",
    )
    fig2_11
    return ax11, ax_list_11, df11, fig2_11, i11, line11, marker11, x11, y11


@app.cell
def __(mo):
    mo.md(r"""## 図2.18""")
    return


@app.cell
def __(K_list_compare, num_actions_list, plt, result_df_actions_final, result_df_k_final, sns):
    fig2_18, ax_list_18 = plt.subplots(1, 2, figsize=(18, 7), tight_layout=True)
    for i18, df18 in enumerate([result_df_actions_final, result_df_k_final]):
        ax18 = ax_list_18[i18]
        x18 = "num_actions" if i18 == 0 else "K"
        for y18 in ["max_rank_weight", "max_pos_weight", "max_topk_weight"]:
            if y18 == "max_pos_weight":
                marker18 = "X"
                line18 = "--"
            elif y18 == "max_topk_weight":
                marker18 = "^"
                line18 = ":"
            else:
                marker18 = "o"
                line18 = "-"
            sns.lineplot(
                markers=True,
                markersize=25,
                linewidth=7,
                legend=False,
                marker=marker18,
                linestyle=line18,
                x=x18,
                y=y18,
                ax=ax18,
                data=df18,
            )
        # yaxis
        ax18.set_ylabel("")
        ax18.set_ylim(0.0, 585)
        ax18.tick_params(axis="y", labelsize=20)
        ax18.yaxis.set_label_coords(-0.1, 0.5)
        # xaxis
        if i18 == 0:
            ax18.set_xlabel(r"ユニークな行動の数$|\mathcal{A}|$", fontsize=28)
            ax18.set_xticks(num_actions_list)
            ax18.set_xticklabels(num_actions_list, fontsize=22)
        else:
            ax18.set_xlabel(r"ランキングの長さ$K$", fontsize=28)
            ax18.set_xticks(K_list_compare)
            ax18.set_xticklabels(K_list_compare, fontsize=22)
        ax18.xaxis.set_label_coords(0.5, -0.12)
    fig2_18.legend(
        ["ランキングレベルの重要度重み", "ポジションレベルの重要度重み", "トップkに関する重要度重み"],
        fontsize=25,
        bbox_to_anchor=(0.5, 1.05),
        ncol=3,
        loc="center",
    )
    fig2_18
    return ax18, ax_list_18, df18, fig2_18, i18, line18, marker18, x18, y18


@app.cell
def __():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
