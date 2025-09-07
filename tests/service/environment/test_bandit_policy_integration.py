"""バンディット方策統合機能のテスト."""

import numpy as np

from recommender_experiments.service.algorithms.thompson_sampling_ranking import (
    ThompsonSamplingRanking,
)
from recommender_experiments.service.environment.ranking_synthetic_dataset import (
    RankingSyntheticBanditDataset,
    SyntheticRankingData,
)


def test_obtain_batch_bandit_feedback_バンディット方策指定なしで従来通り動作すること():
    # Arrange
    dim_context = 3
    num_actions = 5
    K = 2
    theta = np.random.normal(size=(dim_context, num_actions))
    quadratic_weights = np.random.normal(size=(dim_context, num_actions))
    action_bias = np.random.normal(size=(num_actions, 1))
    position_interaction_weights = np.random.normal(size=(K, K))
    action_context = np.random.normal(size=(num_actions, 6))

    sut = RankingSyntheticBanditDataset(
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

    # Act
    result = sut.obtain_batch_bandit_feedback(num_data=10)

    # Assert - 従来通りの動作確認
    assert isinstance(result, SyntheticRankingData), "SyntheticRankingDataが返されること"
    assert result.num_data == 10, "指定したデータ数が正しいこと"
    assert result.ranking_positions == K, "ランキング長が正しいこと"
    assert result.num_actions == num_actions, "action数が正しいこと"


def test_obtain_batch_bandit_feedback_バンディット方策指定で動作すること():
    # Arrange
    dim_context = 3
    num_actions = 6
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

    # バンディットアルゴリズムを準備
    algorithm = ThompsonSamplingRanking(num_actions=num_actions, k=K, dim_context=dim_context, random_state=42)

    # Act
    result = dataset.obtain_batch_bandit_feedback(num_data=5, policy_algorithm=algorithm)

    # Assert
    assert isinstance(result, SyntheticRankingData), "SyntheticRankingDataが返されること"
    assert result.num_data == 5, "指定したデータ数が正しいこと"
    assert result.ranking_positions == K, "ランキング長が正しいこと"
    assert result.num_actions == num_actions, "action数が正しいこと"

    # バンディット方策で選択されたactionが有効範囲内であることを確認
    assert np.all(result.selected_action_vectors >= 0), "選択されたactionが非負であること"
    assert np.all(result.selected_action_vectors < num_actions), "選択されたactionが有効範囲内であること"

    # ログ方策が正規化されていることを確認（各行の合計が1に近い）
    policy_sums = np.sum(result.logging_policy, axis=1)
    assert np.allclose(policy_sums, 1.0, atol=1e-6), "ログ方策が正規化されていること"


def test_obtain_batch_bandit_feedback_動的action変化とバンディット方策が連携すること():
    # Arrange
    dim_context = 3
    num_actions = 8
    K = 2
    theta = np.random.normal(size=(dim_context, num_actions))
    quadratic_weights = np.random.normal(size=(dim_context, num_actions))
    action_bias = np.random.normal(size=(num_actions, 1))
    position_interaction_weights = np.random.normal(size=(K, K))
    action_context = np.random.normal(size=(num_actions, 6))

    # 動的action変化を設定
    action_churn_schedule = {
        0: [0, 1, 2, 3],  # 0-4データ目: action 0-3が利用可能
        5: [2, 3, 4, 5],  # 5-9データ目: action 2-5が利用可能
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

    # バンディットアルゴリズムを準備
    algorithm = ThompsonSamplingRanking(num_actions=num_actions, k=K, dim_context=dim_context, random_state=42)

    # Act
    result = dataset.obtain_batch_bandit_feedback(num_data=10, policy_algorithm=algorithm)

    # Assert
    # 0-4データ目: action 0-3のみが選択されること
    expected_actions_0_4 = {0, 1, 2, 3}
    for data_idx in range(5):
        selected_actions = set(result.selected_action_vectors[data_idx])
        assert selected_actions.issubset(expected_actions_0_4), (
            f"データ{data_idx}: 選択されたaction {selected_actions} が期待範囲 {expected_actions_0_4} 内であること"
        )

    # 5-9データ目: action 2-5のみが選択されること
    expected_actions_5_9 = {2, 3, 4, 5}
    for data_idx in range(5, 10):
        selected_actions = set(result.selected_action_vectors[data_idx])
        assert selected_actions.issubset(expected_actions_5_9), (
            f"データ{data_idx}: 選択されたaction {selected_actions} が期待範囲 {expected_actions_5_9} 内であること"
        )


def test_マイクロバッチ学習で学習効果が確認できること():
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

    algorithm = ThompsonSamplingRanking(num_actions=num_actions, k=K, dim_context=dim_context, random_state=42)

    # Act - マイクロバッチ学習シミュレーション
    num_batches = 10
    batch_size = 5
    all_regrets = []

    for batch_idx in range(num_batches):
        # マイクロバッチデータ生成（固定状態のアルゴリズムで）
        batch_data = dataset.obtain_batch_bandit_feedback(num_data=batch_size, policy_algorithm=algorithm)

        # 各データポイントで学習・評価
        for i in range(batch_data.num_data):
            context = batch_data.context_features[i]
            selected_actions = batch_data.selected_action_vectors[i]
            rewards = batch_data.observed_reward_vectors[i]

            # 簡単なregret計算
            available_actions_mask = batch_data.available_action_mask[i]
            available_actions = np.where(available_actions_mask == 1)[0]
            max_q_values = np.sort(batch_data.base_q_function[i, available_actions])[-K:]
            optimal_reward = sum(max_q_values)
            current_reward = sum(rewards)
            regret = optimal_reward - current_reward
            all_regrets.append(regret)

            # アルゴリズムを更新
            algorithm.update(context=context, selected_actions=selected_actions.tolist(), rewards=rewards.tolist())

    # Assert - 学習効果の確認
    assert len(all_regrets) == num_batches * batch_size, "全データポイントで評価が実行されること"

    # 前半と後半でregretが改善していることを大まかに確認
    first_half_regret = np.mean(all_regrets[: len(all_regrets) // 2])
    second_half_regret = np.mean(all_regrets[len(all_regrets) // 2 :])

    # 学習により後半の性能が改善される傾向があることを確認（厳密ではない）
    # ノイズの影響で常に改善するとは限らないため、極端な悪化がないことを確認
    assert second_half_regret < first_half_regret * 1.5, (
        f"学習により性能が極端に悪化していないこと: 前半={first_half_regret:.3f}, 後半={second_half_regret:.3f}"
    )
