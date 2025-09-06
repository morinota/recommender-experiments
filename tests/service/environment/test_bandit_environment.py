"""バンディット環境のテスト."""

import numpy as np

from recommender_experiments.service.environment.ranking_synthetic_dataset import (
    RankingSyntheticBanditDataset,
)
from recommender_experiments.service.environment.bandit_environment import (
    RankingSyntheticEnvironment,
)


def test_RankingSyntheticEnvironment_基本動作が正常に機能すること():
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

    sut = RankingSyntheticEnvironment(dataset, random_state=42)

    # Act & Assert
    trial = 0
    context, available_actions = sut.get_context_and_available_actions(trial)

    # Assert - コンテキストの形状と内容
    assert context.shape == (dim_context,), "コンテキストの次元が正しいこと"
    assert len(available_actions) == num_actions, "全actionが利用可能であること"
    assert np.array_equal(available_actions, np.arange(num_actions)), "利用可能actionが正しいこと"

    # 報酬を取得
    selected_actions = [0, 1]
    rewards = sut.get_rewards(context, selected_actions, trial)

    # Assert - 報酬の形状と内容
    assert len(rewards) == len(selected_actions), "報酬数が選択action数と一致すること"
    assert all(isinstance(r, float) for r in rewards), "報酬が浮動小数点数であること"

    # 最適報酬を計算
    optimal_reward = sut.get_optimal_reward(context, available_actions, K)

    # Assert - 最適報酬
    assert isinstance(optimal_reward, float), "最適報酬が浮動小数点数であること"
    assert optimal_reward >= 0, "最適報酬が非負であること"


def test_RankingSyntheticEnvironment_動的action変化が機能すること():
    # Arrange
    dim_context = 3
    num_actions = 8
    K = 2
    theta = np.random.normal(size=(dim_context, num_actions))
    quadratic_weights = np.random.normal(size=(dim_context, num_actions))
    action_bias = np.random.normal(size=(num_actions, 1))
    position_interaction_weights = np.random.normal(size=(K, K))
    action_context = np.random.normal(size=(num_actions, 6))

    # action_churn_schedule を設定
    action_churn_schedule = {
        0: [0, 1, 2, 3],  # 0-9試行目: action 0-3が利用可能
        10: [2, 3, 4, 5],  # 10-19試行目: action 2-5が利用可能
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

    sut = RankingSyntheticEnvironment(dataset, random_state=42)

    # Act & Assert - 前半期間 (trial 0-9)
    trial = 5
    context, available_actions = sut.get_context_and_available_actions(trial)

    assert set(available_actions) == {0, 1, 2, 3}, "前半期間の利用可能actionが正しいこと"

    # Act & Assert - 後半期間 (trial 10-19)
    trial = 15
    context, available_actions = sut.get_context_and_available_actions(trial)

    assert set(available_actions) == {2, 3, 4, 5}, "後半期間の利用可能actionが正しいこと"


def test_RankingSyntheticEnvironment_再現性が保証されること():
    # Arrange
    dim_context = 2
    num_actions = 4
    K = 2
    theta = np.random.normal(size=(dim_context, num_actions))
    quadratic_weights = np.random.normal(size=(dim_context, num_actions))
    action_bias = np.random.normal(size=(num_actions, 1))
    position_interaction_weights = np.random.normal(size=(K, K))
    action_context = np.random.normal(size=(num_actions, 4))

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

    sut1 = RankingSyntheticEnvironment(dataset, random_state=999)
    sut2 = RankingSyntheticEnvironment(dataset, random_state=999)

    # Act
    trial = 5
    context1, available_actions1 = sut1.get_context_and_available_actions(trial)
    context2, available_actions2 = sut2.get_context_and_available_actions(trial)

    selected_actions = [0, 1]
    rewards1 = sut1.get_rewards(context1, selected_actions, trial)
    rewards2 = sut2.get_rewards(context2, selected_actions, trial)

    # Assert - 同じrandom_stateで同じ結果が得られること
    assert np.array_equal(context1, context2), "同じrandom_stateで同じコンテキストが生成されること"
    assert np.array_equal(available_actions1, available_actions2), "同じrandom_stateで同じ利用可能actionが生成されること"
    assert np.allclose(rewards1, rewards2), "同じrandom_stateで同じ報酬が生成されること"