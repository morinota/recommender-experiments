"""動的候補プール用Contextual Banditデータセットのテスト."""

import numpy as np
import pytest

from recommender_experiments.service.environment.dynamic_contextual_bandit_dataset import (
    CumulativeRewardSimulationData,
    DynamicContextualBanditDataset,
    ItemLifecycle,
)


def test_指定された設定をもとに動的候補プールの合成ログデータが生成されること():
    # Arrange
    dim_context = 5
    max_num_actions = 10
    days = 3
    daily_traffic = 100

    item_lifecycles = [
        ItemLifecycle(item_id=0, start_day=0, duration_days=3),  # 0-2日目まで有効
        ItemLifecycle(item_id=1, start_day=0, duration_days=2),  # 0-1日目まで有効
        ItemLifecycle(item_id=2, start_day=1, duration_days=2),  # 1-2日目まで有効
        ItemLifecycle(item_id=3, start_day=2, duration_days=1),  # 2日目のみ有効
    ]

    sut = DynamicContextualBanditDataset(
        dim_context=dim_context,
        max_num_actions=max_num_actions,
        item_lifecycles=item_lifecycles,
        days=days,
        daily_traffic=daily_traffic,
        reward_noise=0.1,
        random_state=42,
    )

    # Act
    result = sut.generate_simulation_data()

    # Assert
    assert isinstance(result, CumulativeRewardSimulationData), (
        "CumulativeRewardSimulationDataオブジェクトが返されること"
    )
    assert result.days == days, f"指定された日数 {days} が設定されること"
    assert result.daily_traffic == daily_traffic, f"指定された日別traffic数 {daily_traffic} が設定されること"
    assert len(result.daily_data) == days, f"{days}日分のデータが生成されること"

    total_trials = sum(len(daily.context_features) for daily in result.daily_data)
    expected_total_trials = days * daily_traffic
    assert total_trials == expected_total_trials, f"総trial数が {expected_total_trials} になること"


def test_アイテムライフサイクルに従って候補プールが動的に変化すること():
    # Arrange
    item_lifecycles = [
        ItemLifecycle(item_id=0, start_day=0, duration_days=2),  # 0-1日目
        ItemLifecycle(item_id=1, start_day=1, duration_days=2),  # 1-2日目
        ItemLifecycle(item_id=2, start_day=2, duration_days=1),  # 2日目のみ
    ]

    sut = DynamicContextualBanditDataset(
        dim_context=3,
        max_num_actions=5,
        item_lifecycles=item_lifecycles,
        days=3,
        daily_traffic=50,
        random_state=42,
    )

    # Act
    result = sut.generate_simulation_data()

    # Assert
    # 0日目：アイテム0のみ有効
    day0_actions = set(result.daily_data[0].selected_actions)
    assert day0_actions.issubset({0}), "0日目はアイテム0のみが選択可能であること"

    # 1日目：アイテム0,1が有効
    day1_actions = set(result.daily_data[1].selected_actions)
    assert day1_actions.issubset({0, 1}), "1日目はアイテム0,1が選択可能であること"

    # 2日目：アイテム1,2が有効
    day2_actions = set(result.daily_data[2].selected_actions)
    assert day2_actions.issubset({1, 2}), "2日目はアイテム1,2が選択可能であること"


def test_累積報酬が日を跨いで計算されること():
    # Arrange
    item_lifecycles = [
        ItemLifecycle(item_id=0, start_day=0, duration_days=3),
        ItemLifecycle(item_id=1, start_day=0, duration_days=3),
    ]

    sut = DynamicContextualBanditDataset(
        dim_context=2,
        max_num_actions=3,
        item_lifecycles=item_lifecycles,
        days=2,
        daily_traffic=10,
        random_state=42,
    )

    # Act
    result = sut.generate_simulation_data()

    # Assert
    assert len(result.cumulative_rewards) == 2, "日数分の累積報酬が計算されること"

    # 累積報酬は単調増加であること
    assert result.cumulative_rewards[1] >= result.cumulative_rewards[0], "累積報酬が単調増加すること"

    # 各日の報酬が計算されていること（ノイズにより負の値も許容）
    daily_rewards = [result.cumulative_rewards[0]]
    if len(result.cumulative_rewards) > 1:
        daily_rewards.extend(
            [
                result.cumulative_rewards[i] - result.cumulative_rewards[i - 1]
                for i in range(1, len(result.cumulative_rewards))
            ]
        )

    # 全てが極端に小さい値でないことを確認（正常に動作している証拠）
    assert len(daily_rewards) == 2, "2日分の日別報酬が計算されること"
    assert abs(sum(daily_rewards)) > 0, "報酬が0でない値を持つこと（ノイズにより正負どちらでも可）"


def test_traffic数が指定した値と一致すること():
    # Arrange
    daily_traffic = 25
    days = 4

    sut = DynamicContextualBanditDataset(
        dim_context=3,
        max_num_actions=5,
        item_lifecycles=[
            ItemLifecycle(item_id=0, start_day=0, duration_days=days),
        ],
        days=days,
        daily_traffic=daily_traffic,
        random_state=42,
    )

    # Act
    result = sut.generate_simulation_data()

    # Assert
    for i, daily_data in enumerate(result.daily_data):
        assert len(daily_data.context_features) == daily_traffic, f"{i}日目のtraffic数が {daily_traffic} であること"
        assert len(daily_data.selected_actions) == daily_traffic, f"{i}日目の行動数が {daily_traffic} であること"
        assert len(daily_data.observed_rewards) == daily_traffic, f"{i}日目の報酬数が {daily_traffic} であること"


def test_コンテキスト特徴量の次元数が正しく設定されること():
    # Arrange
    dim_context = 7

    sut = DynamicContextualBanditDataset(
        dim_context=dim_context,
        max_num_actions=5,
        item_lifecycles=[
            ItemLifecycle(item_id=0, start_day=0, duration_days=2),
        ],
        days=2,
        daily_traffic=20,
        random_state=42,
    )

    # Act
    result = sut.generate_simulation_data()

    # Assert
    for daily_data in result.daily_data:
        assert daily_data.context_features.shape[1] == dim_context, (
            f"コンテキスト特徴量の次元数が {dim_context} であること"
        )
