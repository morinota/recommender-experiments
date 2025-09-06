import numpy as np

from recommender_experiments.service.environment.ranking_synthetic_dataset import (
    RankingSyntheticBanditDataset,
    SyntheticRankingData,
)


def test_設定をもとにランキングタスクの合成ログデータが生成されること():
    # Arrange
    num_data = 10
    dim_context = 3
    num_actions = 5
    K = 2
    theta = np.random.normal(size=(dim_context, num_actions))
    quadratic_weights = np.random.normal(size=(dim_context, num_actions))
    action_bias = np.random.normal(size=(num_actions, 1))
    position_interaction_weights = np.random.normal(size=(K, K))
    action_context = np.random.normal(size=(num_actions, 6))
    random_state = 12345

    sut = RankingSyntheticBanditDataset(
        dim_context=dim_context,
        num_actions=num_actions,
        k=K,
        theta=theta,
        quadratic_weights=quadratic_weights,
        action_bias=action_bias,
        position_interaction_weights=position_interaction_weights,
        action_context=action_context,
        random_state=random_state,
    )

    # Act
    result = sut.obtain_batch_bandit_feedback(num_data)

    # Assert - 返り値の型と基本属性
    assert isinstance(result, SyntheticRankingData)
    assert result.num_data == num_data
    assert result.ranking_positions == K
    assert result.num_actions == num_actions

    # Assert - データの形状
    assert result.context_features.shape == (num_data, dim_context)
    assert result.selected_action_vectors.shape == (num_data, K)
    assert result.observed_reward_vectors.shape == (num_data, K)
    assert result.user_behavior_matrix.shape == (num_data, K, K)
    assert result.logging_policy.shape == (num_data, num_actions)
    assert result.expected_rewards.shape == (num_data, K)
    assert result.base_q_function.shape == (num_data, num_actions)

    # Assert - データの値域制約
    assert np.all(result.user_behavior_matrix >= 0) and np.all(
        result.user_behavior_matrix <= 1
    )  # ユーザ行動行列は[0,1]
    assert np.all(result.selected_action_vectors >= 0) and np.all(
        result.selected_action_vectors < num_actions
    )  # 行動は有効範囲内
    assert np.allclose(result.logging_policy.sum(axis=1), 1.0, atol=1e-6)  # 方策は確率分布
    assert np.all(result.base_q_function >= 0) and np.all(result.base_q_function <= 1)  # sigmoid出力


def test_再現性_同じrandom_stateを使用した場合に同じ結果が得られること():
    # Arrange
    num_data = 5
    dim_context = 2
    num_actions = 3
    K = 2
    theta = np.random.normal(size=(dim_context, num_actions))
    quadratic_weights = np.random.normal(size=(dim_context, num_actions))
    action_bias = np.random.normal(size=(num_actions, 1))
    position_interaction_weights = np.random.normal(size=(K, K))
    action_context = np.random.normal(size=(num_actions, 4))
    random_state = 999

    dataset_params = {
        "dim_context": dim_context,
        "num_actions": num_actions,
        "k": K,
        "theta": theta,
        "quadratic_weights": quadratic_weights,
        "action_bias": action_bias,
        "position_interaction_weights": position_interaction_weights,
        "action_context": action_context,
        "random_state": random_state,
    }

    sut1 = RankingSyntheticBanditDataset(**dataset_params)
    sut2 = RankingSyntheticBanditDataset(**dataset_params)

    # Act
    result1 = sut1.obtain_batch_bandit_feedback(num_data)
    result2 = sut2.obtain_batch_bandit_feedback(num_data)

    # Assert - すべてのデータが一致
    assert np.array_equal(result1.context_features, result2.context_features)
    assert np.array_equal(result1.selected_action_vectors, result2.selected_action_vectors)
    assert np.array_equal(result1.observed_reward_vectors, result2.observed_reward_vectors)
    assert np.array_equal(result1.user_behavior_matrix, result2.user_behavior_matrix)
    assert np.array_equal(result1.logging_policy, result2.logging_policy)
    assert np.array_equal(result1.expected_rewards, result2.expected_rewards)
    assert np.array_equal(result1.base_q_function, result2.base_q_function)


# TODO: 将来的に確率的なaction除外が必要になったら有効化
# def test_動的action変化_利用可能性率に基づいてactionが選択されること():
#     # Arrange
#     num_data = 100  # 十分なデータ数でテスト
#     dim_context = 3
#     num_actions = 10
#     K = 2
#     theta = np.random.normal(size=(dim_context, num_actions))
#     quadratic_weights = np.random.normal(size=(dim_context, num_actions))
#     action_bias = np.random.normal(size=(num_actions, 1))
#     position_interaction_weights = np.random.normal(size=(K, K))
#     action_context = np.random.normal(size=(num_actions, 6))

#     # action_availability_rate を設定（50%の確率でactionが利用可能）
#     action_availability_rate = 0.5

#     sut = RankingSyntheticBanditDataset(
#         dim_context=dim_context,
#         num_actions=num_actions,
#         k=K,
#         theta=theta,
#         quadratic_weights=quadratic_weights,
#         action_bias=action_bias,
#         position_interaction_weights=position_interaction_weights,
#         action_context=action_context,
#         action_availability_rate=action_availability_rate,
#         random_state=42,
#     )

#     # Act
#     result = sut.obtain_batch_bandit_feedback(num_data)

#     # Assert
#     assert hasattr(result, "available_action_mask"), "available_action_maskが存在すること"
#     assert result.available_action_mask.shape == (num_data, num_actions), "マスクの形状が正しいこと"
#     assert np.all((result.available_action_mask == 0) | (result.available_action_mask == 1)), (
#         "マスクは0または1の値であること"
#     )

#     # 利用可能性率がおおよそ設定値に近いことを確認
#     availability_ratio = np.mean(result.available_action_mask)
#     assert abs(availability_ratio - action_availability_rate) < 0.1, (
#         f"利用可能性率が設定値に近いこと: {availability_ratio}"
#     )

#     # 選択されたactionが利用可能なactionの範囲内であることを確認
#     for i in range(num_data):
#         available_actions = np.where(result.available_action_mask[i] == 1)[0]
#         for k in range(K):
#             selected_action = result.selected_action_vectors[i, k]
#             assert selected_action in available_actions, (
#                 f"選択されたaction {selected_action} が利用可能なactionの範囲内であること"
#             )


def test_動的action変化_時間軸でのaction入れ替わりが機能すること():
    # Arrange
    num_data = 20
    dim_context = 3
    num_actions = 8
    K = 2
    theta = np.random.normal(size=(dim_context, num_actions))
    quadratic_weights = np.random.normal(size=(dim_context, num_actions))
    action_bias = np.random.normal(size=(num_actions, 1))
    position_interaction_weights = np.random.normal(size=(K, K))
    action_context = np.random.normal(size=(num_actions, 6))

    # action_churn_schedule を設定（10データごとにactionが入れ替わる）
    action_churn_schedule = {
        0: [0, 1, 2, 3],  # 0-9データ目: action 0-3が利用可能
        10: [2, 3, 4, 5],  # 10-19データ目: action 2-5が利用可能
    }

    sut = RankingSyntheticBanditDataset(
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

    # Act
    result = sut.obtain_batch_bandit_feedback(num_data)

    # Assert
    # 0-9データ目: action 0-3のみが利用可能
    expected_actions_0_9 = {0, 1, 2, 3}
    for trial_idx in range(10):
        available_actions = np.where(result.available_action_mask[trial_idx] == 1)[0]
        assert set(available_actions) == expected_actions_0_9, (
            f"データ{trial_idx}: 期待されるactionセットが利用可能であること"
        )

        # 実際に選択された行動が利用可能なactionの範囲内であることを確認
        for position_idx in range(K):
            selected_action = result.selected_action_vectors[trial_idx, position_idx]
            assert selected_action in expected_actions_0_9, (
                f"データ{trial_idx}, ポジション{position_idx}: 選択されたaction {selected_action} が期待されるactionセット {expected_actions_0_9} 内であること"
            )

    # 10-19データ目: action 2-5のみが利用可能
    expected_actions_10_19 = {2, 3, 4, 5}
    for trial_idx in range(10, 20):
        available_actions = np.where(result.available_action_mask[trial_idx] == 1)[0]
        assert set(available_actions) == expected_actions_10_19, (
            f"データ{trial_idx}: 期待されるactionセットが利用可能であること"
        )

        # 実際に選択された行動が利用可能なactionの範囲内であることを確認
        for position_idx in range(K):
            selected_action = result.selected_action_vectors[trial_idx, position_idx]
            assert selected_action in expected_actions_10_19, (
                f"データ{trial_idx}, ポジション{position_idx}: 選択されたaction {selected_action} が期待されるactionセット {expected_actions_10_19} 内であること"
            )
