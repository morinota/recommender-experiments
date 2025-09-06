"""オンラインバンディット学習シミュレータのテスト."""

import numpy as np
import pytest

from recommender_experiments.service.algorithms.bandit_algorithm_interface import (
    BanditAlgorithmInterface,
    OnlineEvaluationResults,
)
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


def test_オンライン学習シミュレータが正常に動作すること():
    # Arrange
    dim_context = 3
    num_actions = 5
    K = 2
    theta = np.random.normal(size=(dim_context, num_actions))
    quadratic_weights = np.random.normal(size=(dim_context, num_actions))
    action_bias = np.random.normal(size=(num_actions, 1))
    position_interaction_weights = np.random.normal(size=(K, K))
    action_context = np.random.normal(size=(num_actions, 6))

    dataset = RankingSyntheticBanditDataset(
        dim_context=dim_context,
        num_actions=num_actions,
        k=K,
        theta=theta,
        quadratic_weights=quadratic_weights,
        action_bias=action_bias,
        position_interaction_weights=position_interaction_weights,
        action_context=action_context,
        random_state=42,
    )

    environment = RankingSyntheticEnvironment(dataset, random_state=42)
    simulator = OnlineBanditSimulator(environment)
    algorithm = ThompsonSamplingRanking(num_actions=num_actions, k=K, dim_context=dim_context, random_state=42)

    num_trials = 100

    # Act
    results = simulator.evaluate_online_learning(
        algorithm=algorithm,
        num_trials=num_trials,
    )

    # Assert
    assert isinstance(results, OnlineEvaluationResults), "OnlineEvaluationResultsオブジェクトが返されること"
    assert results.num_trials == num_trials, f"試行数が {num_trials} であること"
    assert len(results.cumulative_regret) == num_trials, f"累積regretが {num_trials} 要素であること"
    assert len(results.instant_regret) == num_trials, f"瞬時regretが {num_trials} 要素であること"
    assert len(results.cumulative_reward) == num_trials, f"累積報酬が {num_trials} 要素であること"
    assert len(results.selected_actions_history) == num_trials, f"行動履歴が {num_trials} 要素であること"

    # 累積値は単調増加（ノイズにより負の報酬もあるため、累積報酬の最終値が最初より大きいことを確認）
    assert results.cumulative_reward[-1] >= results.cumulative_reward[0], "最終累積報酬が初期値以上であること"

    # regretは期待値的には非負（ノイズにより一部負の値もありうる）
    average_regret = np.mean(results.instant_regret)
    assert average_regret >= -1.0, f"平均regretが極端に負でないこと: {average_regret}"  # ノイズを考慮した緩い条件


def test_動的action変化環境でのオンライン学習が機能すること():
    # Arrange
    dim_context = 3
    num_actions = 6
    K = 2
    theta = np.random.normal(size=(dim_context, num_actions))
    quadratic_weights = np.random.normal(size=(dim_context, num_actions))
    action_bias = np.random.normal(size=(num_actions, 1))
    position_interaction_weights = np.random.normal(size=(K, K))
    action_context = np.random.normal(size=(num_actions, 6))

    # action_churn_schedule を設定
    action_churn_schedule = {
        0: [0, 1, 2],  # 0-49試行目: action 0-2が利用可能
        50: [1, 2, 3, 4],  # 50-99試行目: action 1-4が利用可能
    }

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
    simulator = OnlineBanditSimulator(environment)
    algorithm = ThompsonSamplingRanking(num_actions=num_actions, k=K, dim_context=dim_context, random_state=42)

    num_trials = 100

    # Act
    results = simulator.evaluate_online_learning(algorithm=algorithm, num_trials=num_trials)

    # Assert
    # 動的変化環境でもシミュレーションが正常完了すること
    assert len(results.cumulative_regret) == num_trials, "動的環境で全試行が完了すること"

    # 0-49試行目と50-99試行目で選択された行動が適切な範囲内であること
    first_period_actions = set()
    second_period_actions = set()

    for i in range(50):
        for k in range(K):
            action = results.selected_actions_history[i][k]
            first_period_actions.add(action)

    for i in range(50, 100):
        for k in range(K):
            action = results.selected_actions_history[i][k]
            second_period_actions.add(action)

    assert first_period_actions.issubset({0, 1, 2}), f"前半期間の行動が期待範囲内であること: {first_period_actions}"
    assert second_period_actions.issubset({1, 2, 3, 4}), (
        f"後半期間の行動が期待範囲内であること: {second_period_actions}"
    )


def test_複数のアルゴリズムの性能比較ができること():
    # Arrange
    dim_context = 2
    num_actions = 4
    K = 2
    theta = np.random.normal(size=(dim_context, num_actions))
    quadratic_weights = np.random.normal(size=(dim_context, num_actions))
    action_bias = np.random.normal(size=(num_actions, 1))
    position_interaction_weights = np.random.normal(size=(K, K))
    action_context = np.random.normal(size=(num_actions, 6))

    dataset = RankingSyntheticBanditDataset(
        dim_context=dim_context,
        num_actions=num_actions,
        k=K,
        theta=theta,
        quadratic_weights=quadratic_weights,
        action_bias=action_bias,
        position_interaction_weights=position_interaction_weights,
        action_context=action_context,
        random_state=42,
    )

    environment = RankingSyntheticEnvironment(dataset, random_state=42)
    simulator = OnlineBanditSimulator(environment)

    algorithms = {
        "thompson_sampling": ThompsonSamplingRanking(
            num_actions=num_actions, k=K, dim_context=dim_context, random_state=42
        ),
        # 他のアルゴリズムも追加可能
    }

    num_trials = 50

    # Act
    comparison_results = simulator.compare_algorithms(algorithms=algorithms, num_trials=num_trials)

    # Assert
    assert "thompson_sampling" in comparison_results, "Thompson Samplingの結果が含まれること"
    assert len(comparison_results["thompson_sampling"].cumulative_regret) == num_trials, "試行数が正しいこと"
    assert len(comparison_results["thompson_sampling"].cumulative_reward) == num_trials, "累積報酬が記録されること"
