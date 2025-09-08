import marimo

__generated_with = "0.10.6"
app = marimo.App(width="medium")


@app.cell
def __():
    mo.md("# 6.1 方策の長期性能に関するオフライン評価")
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
    y_label_dict = {"se": "平均二乗誤差", "bias": "二乗バイアス", "variance": "バリアンス", "selection": "方策選択"}

    from dataset import calc_true_value, generate_synthetic_data
    from estimators import calc_ips, calc_new, calc_online
    from utils import aggregate_simulation_results, eps_greedy_policy, softmax_policy

    return (
        DataFrame,
        aggregate_simulation_results,
        calc_ips,
        calc_new,
        calc_online,
        calc_true_value,
        check_random_state,
        eps_greedy_policy,
        generate_synthetic_data,
        japanize_matplotlib,
        np,
        pd,
        plt,
        sns,
        softmax_policy,
        tqdm,
        warnings,
        y_label_dict,
    )


@app.cell
def __():
    mo.md("### ログデータのサイズ$n$を変化させたときの推定量の挙動")
    return


@app.cell
def __(check_random_state):
    ## シミュレーション設定
    num_runs = 50  # シミュレーションの繰り返し回数
    dim_context = 10  # 特徴量xの次元
    num_data = 500  # ログデータのサイズ
    num_actions = 4  # 行動数, |A|
    T = 12  # 総時点数
    eps = 0.0  # データ収集方策のパラメータ, これは共通サポートの仮定を満たさない
    beta = -5  # 評価方策のパラメータ
    random_state = 12345
    random_ = check_random_state(random_state)
    num_data_list = [250, 500, 1000, 2000, 4000]  # ログデータのサイズ
    return T, beta, dim_context, eps, num_actions, num_data, num_data_list, num_runs, random_, random_state


@app.cell
def __(
    aggregate_simulation_results,
    beta,
    calc_ips,
    calc_new,
    calc_online,
    calc_true_value,
    check_random_state,
    dim_context,
    eps,
    generate_synthetic_data,
    num_actions,
    num_data_list,
    num_runs,
    pd,
    random_state,
    softmax_policy,
    T,
    tqdm,
):
    result_df_list = []
    for num_data in num_data_list:
        ## 期待報酬関数を定義するためのパラメータを抽出
        random_ = check_random_state(random_state)
        theta = random_.normal(size=(dim_context, num_actions))
        M = random_.normal(size=(dim_context, num_actions))
        b = random_.normal(size=(1, num_actions))
        W = random_.uniform(0, 1, size=(T, T))
        ## データ収集方策と評価方策の真の性能(policy value)を近似
        policy_value_of_pi0, policy_value_of_pi = calc_true_value(
            dim_context=dim_context,
            num_actions=num_actions,
            theta=theta,
            M=M,
            b=b,
            W=W,
            T=T,
            beta=beta,
            eps=eps,
        )

        estimated_policy_value_list, selection_result_list = [], []
        for _ in tqdm(range(num_runs), desc=f"num_data={num_data}..."):
            ## データ収集方策が形成する分布に従いログデータを生成
            offline_logged_data = generate_synthetic_data(
                num_data=num_data,
                dim_context=dim_context,
                num_actions=num_actions,
                theta=theta,
                M=M,
                b=b,
                W=W,
                T=T,
                eps=eps,
                random_state=_,
            )
            online_experiment_data = generate_synthetic_data(
                num_data=num_data,
                dim_context=dim_context,
                num_actions=num_actions,
                theta=theta,
                M=M,
                b=b,
                W=W,
                T=1,
                beta=beta,
                is_online=True,
                random_state=_,
            )

            ## ログデータ上における評価方策の行動選択確率を計算
            pi = softmax_policy(beta * offline_logged_data["base_q_func"])

            ## ログデータを用いてオフ方策評価を実行する
            estimated_policy_values, selection_result = dict(), dict()
            V_hat_online, selection_result_online = calc_online(online_experiment_data)
            estimated_policy_values["online"] = V_hat_online
            selection_result["online"] = selection_result_online
            V_hat_ips, selection_result_ips = calc_ips(offline_logged_data, pi)
            estimated_policy_values["ips"] = V_hat_ips
            selection_result["ips"] = selection_result_ips
            V_hat_new, selection_result_new = calc_new(offline_logged_data, online_experiment_data, pi)
            estimated_policy_values["new"] = V_hat_new
            selection_result["new"] = selection_result_new
            estimated_policy_value_list.append(estimated_policy_values)
            selection_result_list.append(selection_result)

        ## シミュレーション結果を集計する
        result_df_list.append(
            aggregate_simulation_results(
                estimated_policy_value_list,
                selection_result_list,
                policy_value_of_pi,
                "num_data",
                num_data,
            )
        )
    result_df_data = pd.concat(result_df_list).reset_index(level=0)
    return (
        M,
        W,
        b,
        estimated_policy_value_list,
        estimated_policy_values,
        num_data,
        offline_logged_data,
        online_experiment_data,
        pi,
        policy_value_of_pi,
        policy_value_of_pi0,
        random_,
        result_df_data,
        result_df_list,
        selection_result,
        selection_result_ips,
        selection_result_list,
        selection_result_new,
        selection_result_online,
        theta,
        V_hat_ips,
        V_hat_new,
        V_hat_online,
    )


@app.cell
def __():
    mo.md("## 図6.4")
    return


@app.cell
def __(plt, result_df_data, sns, y_label_dict):
    fig, ax_list = plt.subplots(1, 4, figsize=(46, 10), tight_layout=True)
    for i, y in enumerate(["se", "bias", "variance", "selection"]):
        ax = ax_list[i]
        sns.lineplot(
            markers=True,
            markersize=40,
            linewidth=12,
            legend=False,
            style="est",
            x="num_data",
            y=y,
            hue="est",
            ax=ax,
            ci=None,
            palette=["tab:grey", "tab:red", "tab:purple"],
            data=result_df_data,
        )
        ax.set_title(y_label_dict[y], fontsize=58)
        # yaxis
        ax.set_ylabel("")
        if i < 3:
            ax.set_ylim(0, 0.024)
            ax.set_yticks([0.0, 0.01, 0.02])
        else:
            ax.set_yscale("linear")
            ax.set_ylim(0.0, 1.05)
        ax.tick_params(axis="y", labelsize=35)
        ax.yaxis.set_label_coords(-0.1, 0.5)
        # xaxis
        ax.set_xlabel(r"ログデータのサイズ$n$", fontsize=50)
        ax.set_xticks([250, 1000, 2000, 4000])
        ax.set_xticklabels([250, 1000, 2000, 4000], fontsize=35)
        ax.xaxis.set_label_coords(0.5, -0.12)
    fig.legend(["AVG", "IPS", "New"], fontsize=65, bbox_to_anchor=(0.5, 1.15), ncol=5, loc="center")
    plt.show()
    return ax, ax_list, fig, i, y


@app.cell
def __():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
