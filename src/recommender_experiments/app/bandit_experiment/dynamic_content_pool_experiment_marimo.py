"""動的コンテンツプールでのトンプソンサンプリング性能実験（marimo最適化版）"""

import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # 動的コンテンツプールでのトンプソンサンプリング性能実験

    このノートブックでは、推薦候補コンテンツプールが動的に変化する環境において、
    Context-free Thompson SamplingとContextual Thompson Samplingの性能を比較します。

    ## 実験概要
    - **動的コンテンツプール**: 実験期間中にactionセットが3段階で変化
    - **アルゴリズム比較**: Context-free vs Contextual Thompson Sampling
    - **評価指標**: 累積報酬、瞬時報酬、段階別性能分析
    """
    )
    return


@app.cell
def _(mo):
    # ライブラリのインポート
    from typing import Dict, List, Tuple

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    from recommender_experiments.service.algorithms.bandit_algorithm_interface import OnlineEvaluationResults
    from recommender_experiments.service.algorithms.thompson_sampling_contextfree import ThompsonSamplingContextFree
    from recommender_experiments.service.algorithms.thompson_sampling_ranking import ThompsonSamplingRanking
    from recommender_experiments.service.environment.ranking_synthetic_dataset import RankingSyntheticBanditDataset

    # 可視化の設定
    plt.style.use("seaborn-whitegrid")
    sns.set_palette("husl")
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.rcParams["font.size"] = 10

    mo.md("✅ ライブラリのインポート完了")
    return (
        Dict,
        List,
        OnlineEvaluationResults,
        RankingSyntheticBanditDataset,
        ThompsonSamplingContextFree,
        ThompsonSamplingRanking,
        np,
        pd,
        plt,
    )


@app.cell
def _(mo):
    # 実験設定のUIコンポーネント
    mo.md("## 🎛️ 実験設定")
    return


@app.cell
def _(mo):
    # インタラクティブな実験設定
    num_trials_slider = mo.ui.slider(start=100, stop=2000, step=100, value=1000, label="実験試行数")

    num_actions_initial_slider = mo.ui.slider(start=10, stop=30, step=5, value=20, label="初期action数")

    num_actions_total_slider = mo.ui.slider(start=30, stop=100, step=10, value=50, label="最大action数")

    k_slider = mo.ui.slider(start=2, stop=5, step=1, value=3, label="ランキング長")

    dim_context_slider = mo.ui.slider(start=3, stop=10, step=1, value=5, label="コンテキスト次元")

    settings_form = mo.vstack(
        [num_trials_slider, num_actions_initial_slider, num_actions_total_slider, k_slider, dim_context_slider]
    )

    return (
        dim_context_slider,
        k_slider,
        num_actions_initial_slider,
        num_actions_total_slider,
        num_trials_slider,
        settings_form,
    )


@app.cell
def _(settings_form):
    settings_form
    return


@app.cell
def _(
    dim_context_slider,
    k_slider,
    mo,
    num_actions_initial_slider,
    num_actions_total_slider,
    num_trials_slider,
):
    # 設定値の取得と表示
    RANDOM_STATE = 12345
    NUM_TRIALS = num_trials_slider.value
    NUM_ACTIONS_INITIAL = num_actions_initial_slider.value
    NUM_ACTIONS_TOTAL = num_actions_total_slider.value
    K = k_slider.value
    DIM_CONTEXT = dim_context_slider.value

    config_display = mo.md(f"""
    **現在の実験設定:**
    - 実験試行数: {NUM_TRIALS}
    - 初期action数: {NUM_ACTIONS_INITIAL} 
    - 最大action数: {NUM_ACTIONS_TOTAL}
    - ランキング長: {K}
    - コンテキスト次元: {DIM_CONTEXT}
    - 乱数シード: {RANDOM_STATE}
    """)

    config_display
    return (
        DIM_CONTEXT,
        K,
        NUM_ACTIONS_INITIAL,
        NUM_ACTIONS_TOTAL,
        NUM_TRIALS,
        RANDOM_STATE,
    )


@app.cell
def _(Dict, List, NUM_ACTIONS_INITIAL, NUM_ACTIONS_TOTAL, NUM_TRIALS):
    # action変更スケジュール作成
    def create_action_churn_schedule(
        num_trials: int, num_actions_initial: int, num_actions_total: int
    ) -> Dict[int, List[int]]:
        """動的にコンテンツプールが変化するスケジュールを作成する"""
        schedule = {}

        # 第1段階: 初期のactionセット (0 - num_trials//3)
        stage1_end = num_trials // 3
        schedule[0] = list(range(num_actions_initial))

        # 第2段階: 一部actionが削除され、新しいactionが追加 (num_trials//3 - 2*num_trials//3)
        stage2_end = 2 * num_trials // 3
        remaining_initial = list(range(num_actions_initial // 2, num_actions_initial))
        new_actions = list(range(num_actions_initial, num_actions_initial + 10))
        schedule[stage1_end] = remaining_initial + new_actions

        # 第3段階: さらに多くの新しいactionが追加される (2*num_trials//3 - num_trials)
        more_new_actions = list(range(num_actions_initial + 10, num_actions_total))
        schedule[stage2_end] = remaining_initial + new_actions + more_new_actions

        return schedule

    action_churn_schedule = create_action_churn_schedule(NUM_TRIALS, NUM_ACTIONS_INITIAL, NUM_ACTIONS_TOTAL)

    return (action_churn_schedule,)


@app.cell
def _(action_churn_schedule, mo, pd):
    # スケジュールの表示
    schedule_info = []
    for trial_start, actions in action_churn_schedule.items():
        schedule_stage_end = min(
            [t for t in action_churn_schedule.keys() if t > trial_start]
            + [list(action_churn_schedule.keys())[-1] + 100]
        )
        schedule_info.append(
            {
                "開始Trial": trial_start,
                "Action数": len(actions),
                "Action例": str(actions[:5]) + ("..." if len(actions) > 5 else ""),
            }
        )

    schedule_df = pd.DataFrame(schedule_info)

    mo.vstack([mo.md("## 📅 Action変更スケジュール"), mo.ui.table(schedule_df)])
    return


@app.cell
def _(
    DIM_CONTEXT,
    K,
    NUM_ACTIONS_TOTAL,
    RANDOM_STATE,
    RankingSyntheticBanditDataset,
    action_churn_schedule,
    np,
):
    # データセット環境作成
    def create_dataset_environment(
        action_churn_schedule_param: dict, num_actions_total: int
    ) -> RankingSyntheticBanditDataset:
        """実験用のデータセット環境を作成する"""
        np.random.seed(RANDOM_STATE)

        # action特徴量の生成
        action_context = np.random.randn(num_actions_total, DIM_CONTEXT)

        # 期待報酬関数のパラメータを設定
        theta = np.random.randn(DIM_CONTEXT, num_actions_total) * 0.5
        quadratic_weights = np.random.randn(DIM_CONTEXT, num_actions_total) * 0.2
        action_bias = np.random.randn(num_actions_total, 1) * 0.1
        position_interaction_weights = np.random.randn(K, K) * 0.1

        # データセット環境を作成
        dataset_env = RankingSyntheticBanditDataset(
            dim_context=DIM_CONTEXT,
            num_actions=num_actions_total,
            k=K,
            action_context=action_context,
            theta=theta,
            quadratic_weights=quadratic_weights,
            action_bias=action_bias,
            position_interaction_weights=position_interaction_weights,
            beta=1.0,  # softmax温度パラメータ
            reward_noise=0.1,
            random_state=RANDOM_STATE,
            action_churn_schedule=action_churn_schedule_param,
        )

        return dataset_env

    dataset_env = create_dataset_environment(action_churn_schedule, NUM_ACTIONS_TOTAL)
    return (dataset_env,)


@app.cell
def _(
    Dict,
    K,
    List,
    OnlineEvaluationResults,
    RankingSyntheticBanditDataset,
    np,
):
    # 実験実行関数
    def run_online_bandit_experiment(
        dataset_env: RankingSyntheticBanditDataset,
        algorithm_name: str,
        algorithm_instance,
        num_trials: int,
        action_churn_schedule_param: Dict[int, List[int]],
    ) -> OnlineEvaluationResults:
        """オンラインバンディット実験を実行する"""
        results = OnlineEvaluationResults(algorithm_name)
        algorithm_instance.reset()

        for trial in range(num_trials):
            # 1回分のデータを生成
            synthetic_data = dataset_env.obtain_batch_bandit_feedback(1)

            context = synthetic_data.context_features[0]
            available_actions = np.where(synthetic_data.available_action_mask[0] == 1)[0]

            # アルゴリズムでaction選択
            selected_actions = algorithm_instance.select_actions(context, available_actions, K)

            # 最適actionと報酬計算
            available_q_values = synthetic_data.base_q_function[0, available_actions]
            optimal_actions_idx = np.argsort(available_q_values)[-K:][::-1]
            optimal_actions = available_actions[optimal_actions_idx]

            # 実際の報酬取得
            selected_rewards = [
                synthetic_data.base_q_function[0, action_id]
                for action_id in selected_actions
                if action_id < len(synthetic_data.base_q_function[0])
            ]
            optimal_rewards = [
                synthetic_data.base_q_function[0, action_id] for i, action_id in enumerate(optimal_actions) if i < K
            ]

            # regret計算
            instant_reward = sum(selected_rewards) if selected_rewards else 0.0
            optimal_reward_sum = sum(optimal_rewards) if optimal_rewards else 0.0
            instant_regret = optimal_reward_sum - instant_reward

            # 学習更新
            algorithm_instance.update(context, selected_actions, selected_rewards)

            # 結果記録
            results.add_trial_result(selected_actions, instant_regret, instant_reward)

            # 進捗表示
            if (trial + 1) % 200 == 0:
                print(f"  Trial {trial + 1}: 平均報酬 = {results.get_average_reward():.4f}")

        return results

    return (run_online_bandit_experiment,)


@app.cell
def _(mo):
    # 実験実行開始
    mo.md("## 🧪 実験実行")
    return


@app.cell
def _(
    DIM_CONTEXT,
    K,
    NUM_ACTIONS_TOTAL,
    NUM_TRIALS,
    RANDOM_STATE,
    ThompsonSamplingContextFree,
    ThompsonSamplingRanking,
    action_churn_schedule,
    dataset_env,
    mo,
    run_online_bandit_experiment,
):
    # 実験実行開始
    mo.md("⏳ 実験実行中...")

    print("🧪 実験開始")

    # Context-free Thompson Sampling
    print("🚀 Context-free Thompson Sampling実験開始...")
    ts_contextfree = ThompsonSamplingContextFree(
        num_actions=NUM_ACTIONS_TOTAL, k=K, alpha=1.0, beta=1.0, random_state=RANDOM_STATE
    )

    results_contextfree = run_online_bandit_experiment(
        dataset_env, "Thompson Sampling (Context-free)", ts_contextfree, NUM_TRIALS, action_churn_schedule
    )
    print(f"✅ Context-free完了: 累積報酬 {results_contextfree.get_final_cumulative_reward():.2f}")

    # Contextual Thompson Sampling
    print("🚀 Contextual Thompson Sampling実験開始...")
    ts_contextual = ThompsonSamplingRanking(
        num_actions=NUM_ACTIONS_TOTAL, k=K, dim_context=DIM_CONTEXT, alpha=1.0, beta=1.0, random_state=RANDOM_STATE
    )

    results_contextual = run_online_bandit_experiment(
        dataset_env, "Thompson Sampling (Contextual)", ts_contextual, NUM_TRIALS, action_churn_schedule
    )
    print(f"✅ Contextual完了: 累積報酬 {results_contextual.get_final_cumulative_reward():.2f}")

    all_results = [results_contextfree, results_contextual]

    print("✅ 実験完了!")
    mo.md("✅ 実験完了!")
    return all_results, results_contextfree, results_contextual


@app.cell
def _(all_results, mo, pd):
    # 結果サマリー表示
    summary_data = []
    for summary_result in all_results:
        summary_data.append(
            {
                "アルゴリズム": summary_result.algorithm_name,
                "最終累積報酬": f"{summary_result.get_final_cumulative_reward():.2f}",
                "平均報酬": f"{summary_result.get_average_reward():.4f}",
                "平均Regret": f"{summary_result.get_average_regret():.4f}",
            }
        )

    summary_df = pd.DataFrame(summary_data)

    mo.vstack([mo.md("## 📊 実験結果サマリー"), mo.ui.table(summary_df)])
    return


@app.cell
def _(action_churn_schedule, all_results, pd, plt):
    # メイン結果可視化
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 累積報酬の推移 (主要指標)
    ax1 = axes[0, 0]
    for plot_result in all_results:
        ax1.plot(plot_result.cumulative_reward, label=plot_result.algorithm_name, alpha=0.8, linewidth=2)
    ax1.set_xlabel("Trial")
    ax1.set_ylabel("Cumulative Reward")
    ax1.set_title("累積報酬の推移")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 累積Regretの推移 (参考指標)
    ax2 = axes[0, 1]
    for plot_result in all_results:
        ax2.plot(plot_result.cumulative_regret, label=plot_result.algorithm_name, alpha=0.8, linewidth=2)
    ax2.set_xlabel("Trial")
    ax2.set_ylabel("Cumulative Regret")
    ax2.set_title("累積Regretの推移 (参考)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 瞬時Regretの移動平均
    ax3 = axes[1, 0]
    window_size = 50
    for plot_result in all_results:
        instant_regret_series = pd.Series(plot_result.instant_regret)
        moving_avg = instant_regret_series.rolling(window=window_size, center=True).mean()
        ax3.plot(moving_avg, label=f"{plot_result.algorithm_name} (MA={window_size})", alpha=0.8, linewidth=2)
    ax3.set_xlabel("Trial")
    ax3.set_ylabel("Instant Regret (Moving Average)")
    ax3.set_title("瞬時Regretの移動平均")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. action変更タイミングの可視化
    ax4 = axes[1, 1]

    # action変更タイミングを縦線で表示
    change_points = list(action_churn_schedule.keys())
    colors = ["red", "blue", "green"]
    for i, change_point in enumerate(change_points):
        if i < len(colors):
            ax4.axvline(
                x=change_point,
                color=colors[i],
                linestyle="--",
                alpha=0.7,
                linewidth=2,
                label=f"Stage {i + 1} (Actions: {len(action_churn_schedule[change_point])})",
            )

    # 瞬時Regretをbackgroundとして表示
    for plot_result in all_results:
        instant_regret_series = pd.Series(plot_result.instant_regret)
        moving_avg = instant_regret_series.rolling(window=window_size, center=True).mean()
        ax4.plot(moving_avg, alpha=0.3, linewidth=1, label=f"{plot_result.algorithm_name}")

    ax4.set_xlabel("Trial")
    ax4.set_ylabel("Instant Regret")
    ax4.set_title("Action変更タイミングと性能への影響")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    fig
    return


@app.cell
def _(
    Dict,
    List,
    OnlineEvaluationResults,
    action_churn_schedule,
    all_results,
    mo,
    np,
    pd,
):
    # 段階別性能分析
    def analyze_performance_by_stage(
        results: OnlineEvaluationResults, action_churn_schedule_param: Dict[int, List[int]]
    ) -> pd.DataFrame:
        """段階別の性能を分析する"""
        stages = []
        change_points = sorted(action_churn_schedule_param.keys())

        for i in range(len(change_points)):
            stage_start = change_points[i]
            stage_end = change_points[i + 1] if i + 1 < len(change_points) else len(results.instant_regret)

            stage_regrets = results.instant_regret[stage_start:stage_end]
            stage_rewards = results.instant_reward[stage_start:stage_end]

            stages.append(
                {
                    "Stage": i + 1,
                    "開始Trial": stage_start,
                    "終了Trial": stage_end,
                    "Action数": len(action_churn_schedule_param[stage_start]),
                    "平均報酬": np.mean(stage_rewards) if stage_rewards else 0,
                    "平均Regret": np.mean(stage_regrets) if stage_regrets else 0,
                    "アルゴリズム": results.algorithm_name,
                }
            )

        return pd.DataFrame(stages)

    # 各アルゴリズムの段階別分析
    stage_analysis_list = []
    for stage_result in all_results:
        stage_df = analyze_performance_by_stage(stage_result, action_churn_schedule)
        stage_analysis_list.append(stage_df)

    combined_stage_analysis = pd.concat(stage_analysis_list, ignore_index=True)

    mo.vstack([mo.md("## 📈 段階別性能分析"), mo.ui.table(combined_stage_analysis.round(4))])
    return (combined_stage_analysis,)


@app.cell
def _(combined_stage_analysis, plt):
    # 段階別性能の可視化
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))

    # 段階別平均報酬 (主要指標)
    ax2_1 = axes2[0]
    pivot_reward = combined_stage_analysis.pivot(index="Stage", columns="アルゴリズム", values="平均報酬")
    pivot_reward.plot(kind="bar", ax=ax2_1, alpha=0.8, width=0.7)
    ax2_1.set_xlabel("Stage")
    ax2_1.set_ylabel("Average Reward")
    ax2_1.set_title("段階別平均報酬")
    ax2_1.legend(title="アルゴリズム", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2_1.grid(True, alpha=0.3)
    ax2_1.set_xticklabels([f"Stage {i}" for i in range(1, 4)], rotation=0)

    # 段階別平均Regret (参考指標)
    ax2_2 = axes2[1]
    pivot_regret = combined_stage_analysis.pivot(index="Stage", columns="アルゴリズム", values="平均Regret")
    pivot_regret.plot(kind="bar", ax=ax2_2, alpha=0.8, width=0.7)
    ax2_2.set_xlabel("Stage")
    ax2_2.set_ylabel("Average Regret")
    ax2_2.set_title("段階別平均Regret (参考)")
    ax2_2.legend(title="アルゴリズム", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2_2.grid(True, alpha=0.3)
    ax2_2.set_xticklabels([f"Stage {i}" for i in range(1, 4)], rotation=0)

    plt.tight_layout()
    fig2
    return


@app.cell
def _(
    Dict,
    List,
    OnlineEvaluationResults,
    action_churn_schedule,
    all_results,
    mo,
    pd,
):
    # Action選択パターン分析
    def analyze_action_selection_patterns(
        results: OnlineEvaluationResults, action_churn_schedule_param: Dict[int, List[int]]
    ) -> dict:
        """action選択パターンを分析する"""
        pattern_analysis = {}
        change_points = sorted(action_churn_schedule_param.keys())

        for i in range(len(change_points)):
            stage_start = change_points[i]
            stage_end = change_points[i + 1] if i + 1 < len(change_points) else len(results.selected_actions_history)

            stage_actions = results.selected_actions_history[stage_start:stage_end]
            available_actions = action_churn_schedule_param[stage_start]

            # 各actionの選択頻度を計算
            action_counts = {}
            total_selections = 0

            for trial_actions in stage_actions:
                for action_id in trial_actions:
                    action_counts[action_id] = action_counts.get(action_id, 0) + 1
                    total_selections += 1

            # 利用可能actionのみの選択率を計算
            available_action_rates = {}
            for action_id in available_actions:
                rate = action_counts.get(action_id, 0) / max(total_selections, 1)
                available_action_rates[action_id] = rate

            pattern_analysis[f"Stage_{i + 1}"] = {
                "available_actions": available_actions,
                "action_selection_rates": available_action_rates,
                "diversity_score": len([r for r in available_action_rates.values() if r > 0.01])
                / len(available_actions),
            }

        return pattern_analysis

    # Action選択パターン分析の実行と表示
    pattern_results = []
    for pattern_result in all_results:
        patterns = analyze_action_selection_patterns(pattern_result, action_churn_schedule)
        for stage_name, pattern in patterns.items():
            sorted_rates = sorted(pattern["action_selection_rates"].items(), key=lambda x: x[1], reverse=True)
            top_actions = sorted_rates[:3]  # トップ3のみ表示

            pattern_results.append(
                {
                    "アルゴリズム": pattern_result.algorithm_name,
                    "Stage": stage_name.replace("Stage_", ""),
                    "Diversity Score": f"{pattern['diversity_score']:.3f}",
                    "Top3 Actions": str([(action_id, f"{rate:.3f}") for action_id, rate in top_actions]),
                }
            )

    pattern_df = pd.DataFrame(pattern_results)

    mo.vstack([mo.md("## 🎯 Action選択パターン分析"), mo.ui.table(pattern_df)])
    return


@app.cell
def _(
    DIM_CONTEXT,
    K,
    NUM_TRIALS,
    action_churn_schedule,
    mo,
    results_contextfree,
    results_contextual,
):
    # 最終まとめ
    better_algorithm = (
        "Context-free"
        if results_contextfree.get_final_cumulative_reward() > results_contextual.get_final_cumulative_reward()
        else "Contextual"
    )

    stages_info = ""
    for trial_start_idx, action_ids in action_churn_schedule.items():
        stage_end = min([t for t in action_churn_schedule.keys() if t > trial_start_idx] + [NUM_TRIALS])
        stages_info += f"- Trial {trial_start_idx}-{stage_end - 1}: {len(action_ids)}個のコンテンツ\n"

    final_summary = mo.md(f"""
    ## 🎉 実験結果まとめ

    ### 実験設定
    - **総試行数**: {NUM_TRIALS}
    - **ランキング長**: {K}
    - **コンテキスト次元**: {DIM_CONTEXT}
    - **コンテンツプール変化**:
    {stages_info}

    ### 主要な発見
    - **優秀なアルゴリズム**: {better_algorithm} Thompson Sampling
    - **Context-free**: 累積報酬 {results_contextfree.get_final_cumulative_reward():.2f}
    - **Contextual**: 累積報酬 {results_contextual.get_final_cumulative_reward():.2f}

    ### 考察
    {"- シンプルなContext-freeアプローチが動的環境で頑健性を発揮" if better_algorithm == "Context-free" else "- コンテキスト情報が動的環境での適応に有効"}
    - 両アルゴリズムともコンテンツプール変化に適応
    - 新しいコンテンツが追加される際の探索-活用トレードオフが重要
    """)

    final_summary
    return


if __name__ == "__main__":
    app.run()
