"""オンラインバンディット学習評価の使用例."""

import numpy as np

from recommender_experiments.service.algorithms.thompson_sampling_ranking import (
    ThompsonSamplingRanking,
)
from recommender_experiments.service.environment.bandit_environment import (
    RankingSyntheticEnvironment,
)
from recommender_experiments.service.environment.ranking_synthetic_dataset import (
    RankingSyntheticBanditDataset,
)
from recommender_experiments.service.simulator.online_bandit_simulator import (
    OnlineBanditSimulator,
)


def main():
    """オンラインバンディット学習評価の実行例."""
    print("=== オンラインバンディット学習評価の実行 ===")

    # 1. 合成データセットの設定
    dim_context = 5
    num_actions = 10
    K = 3

    # パラメータをランダムに設定
    np.random.seed(42)
    theta = np.random.normal(size=(dim_context, num_actions))
    quadratic_weights = np.random.normal(size=(dim_context, num_actions))
    action_bias = np.random.normal(size=(num_actions, 1))
    position_interaction_weights = np.random.normal(size=(K, K))
    action_context = np.random.normal(size=(num_actions, dim_context * 2))

    # 動的なaction変化を設定
    action_churn_schedule = {
        0: [0, 1, 2, 3, 4],  # 0-199試行目: action 0-4が利用可能
        200: [2, 3, 4, 5, 6, 7],  # 200-399試行目: action 2-7が利用可能
        400: [4, 5, 6, 7, 8, 9],  # 400-599試行目: action 4-9が利用可能
    }

    # 2. データセットと環境を作成
    dataset = RankingSyntheticBanditDataset(
        dim_context=dim_context,
        num_actions=num_actions,
        k=K,
        theta=theta,
        quadratic_weights=quadratic_weights,
        action_bias=action_bias,
        position_interaction_weights=position_interaction_weights,
        action_context=action_context,
        action_churn_schedule=action_churn_schedule,
        random_state=42,
    )

    environment = RankingSyntheticEnvironment(dataset, random_state=42)

    # 3. シミュレータを作成
    simulator = OnlineBanditSimulator(environment, random_state=42)

    # 4. バンディットアルゴリズムを設定
    algorithms = {
        "Thompson Sampling (α=1.0)": ThompsonSamplingRanking(
            num_actions=num_actions, k=K, dim_context=dim_context, alpha=1.0, beta=1.0, random_state=42
        ),
        "Thompson Sampling (α=0.1)": ThompsonSamplingRanking(
            num_actions=num_actions, k=K, dim_context=dim_context, alpha=0.1, beta=1.0, random_state=43
        ),
    }

    # 5. オンライン学習シミュレーションを実行
    num_trials = 600
    print(f"シミュレーション実行中... ({num_trials} trials)")

    comparison_results = simulator.compare_algorithms(algorithms=algorithms, num_trials=num_trials)

    # 6. 結果を表示
    print("\n=== 評価結果 ===")
    for algorithm_name, results in comparison_results.items():
        print(f"\n[{algorithm_name}]")
        print(f"  最終累積報酬: {results.cumulative_reward[-1]:.3f}")
        print(f"  最終累積Regret: {results.cumulative_regret[-1]:.3f}")
        print(f"  平均瞬時Regret: {np.mean(results.instant_regret):.3f}")

        # 各期間の性能を分析
        first_period_regret = np.mean(results.instant_regret[0:200])
        second_period_regret = np.mean(results.instant_regret[200:400])
        third_period_regret = np.mean(results.instant_regret[400:600])

        print(f"  第1期間 (0-199) 平均Regret: {first_period_regret:.3f}")
        print(f"  第2期間 (200-399) 平均Regret: {second_period_regret:.3f}")
        print(f"  第3期間 (400-599) 平均Regret: {third_period_regret:.3f}")

    print("\n=== シミュレーション完了 ===")

    # 7. Action選択の変化を確認
    print(f"\n=== Action選択の変化確認 ===")
    ts_results = comparison_results["Thompson Sampling (α=1.0)"]

    # 各期間で選択されたactionを集計
    first_period_actions = set()
    second_period_actions = set()
    third_period_actions = set()

    for i in range(200):
        first_period_actions.update(ts_results.selected_actions_history[i])

    for i in range(200, 400):
        second_period_actions.update(ts_results.selected_actions_history[i])

    for i in range(400, 600):
        third_period_actions.update(ts_results.selected_actions_history[i])

    print(f"第1期間に選択されたaction: {sorted(first_period_actions)}")
    print(f"第2期間に選択されたaction: {sorted(second_period_actions)}")
    print(f"第3期間に選択されたaction: {sorted(third_period_actions)}")

    # スケジュールとの整合性確認
    print(f"\naction_churn_schedule:")
    for start_trial, actions in action_churn_schedule.items():
        print(f"  trial {start_trial}以降: {actions}")


if __name__ == "__main__":
    main()
