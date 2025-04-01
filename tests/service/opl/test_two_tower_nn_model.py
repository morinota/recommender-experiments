import numpy as np
import torch

from recommender_experiments.service.opl.two_tower_nn_model import (
    PolicyByTwoTowerModel,
)


def test_TwoTowerモデルが正しく初期化されること():
    # Arrange
    dim_context_features = 200
    dim_action_features = 150
    dim_two_tower_embedding = 100

    # Act
    sut = PolicyByTwoTowerModel(
        dim_context_features,
        dim_action_features,
        dim_two_tower_embedding,
    )

    # Assert
    assert isinstance(sut.context_tower, torch.nn.Sequential)
    assert isinstance(sut.action_tower, torch.nn.Sequential)

    # Context Tower の出力サイズ
    context_output_layer = sut.context_tower[-1]  # コンテキストタワーの最後が埋め込み層
    assert isinstance(context_output_layer, torch.nn.Linear)
    assert (
        context_output_layer.out_features == dim_two_tower_embedding
    ), "Context Tower の最終出力サイズが dim_two_tower_embedding に一致すること"

    # Action Tower の出力サイズ
    action_output_layer = sut.action_tower[-1]  # アクションタワーの最後が埋め込み層
    assert isinstance(action_output_layer, torch.nn.Linear)
    assert (
        action_output_layer.out_features == dim_two_tower_embedding
    ), "Action Tower の最終出力サイズが dim_two_tower_embedding に一致すること"

    # 学習時の損失関数と方策性能の記録
    assert sut.train_losses == []
    assert sut.train_values == []
    assert sut.test_values == []


def test_アクション候補の数が動的に変化してもアクション選択の確率分布の推論ができること():
    # Arrange
    n_rounds = 10
    n_actions = 4
    dim_context_features = 200
    dim_action_features = 150
    dim_two_tower_embedding = 100
    sut = PolicyByTwoTowerModel(
        dim_context_features,
        dim_action_features,
        dim_two_tower_embedding,
    )

    # Act
    action_dist = sut.predict_proba(
        context=np.random.random((n_rounds, dim_context_features)),
        action_context=np.random.random((n_actions, dim_action_features)),
    )

    # Assert
    assert action_dist.shape == (
        n_rounds,
        n_actions,
    ), "出力の形状が(ラウンド数、アクション数)であること"
    assert np.allclose(
        action_dist.sum(axis=1), 1.0
    ), "各ラウンドごとに、確率の総和が1.0"
    assert np.all(0 <= action_dist) and np.all(
        action_dist <= 1
    ), "各アクションの選択確率が0.0以上1.0以下であること"

    # アクション候補の数が変化しても、同一モデルで推論できること
    action_dist = sut.predict_proba(
        context=np.random.random((n_rounds, dim_context_features)),
        action_context=np.random.random((n_actions + 2, dim_action_features)),
    )
    assert action_dist.shape == (
        n_rounds,
        n_actions + 2,
    ), "出力の形状が(ラウンド数、アクション数)であること"
    assert np.allclose(
        action_dist.sum(axis=1), 1.0
    ), "各ラウンドごとに、確率の総和が1.0"
    assert np.all(0 <= action_dist) and np.all(
        action_dist <= 1
    ), "各アクションの選択確率が0以上1以下であること"


def test_バンディットフィードバックデータを元にIPS推定量で勾配ベースのオフ方策学習ができること():
    # Arrange
    n_rounds = 100
    n_actions = 4
    dim_context_features = 200
    dim_action_features = 150
    dim_two_tower_embedding = 100
    off_policy_objective = "ips"
    sut = PolicyByTwoTowerModel(
        dim_context_features,
        dim_action_features,
        dim_two_tower_embedding,
        off_policy_objective=off_policy_objective,
    )
    bandit_feedback_train = {
        "n_rounds": n_rounds,
        "n_actions": n_actions,
        "context": np.random.random((n_rounds, dim_context_features)),
        "action_context": np.random.random((n_actions, dim_action_features)),
        "action": np.random.randint(0, n_actions, n_rounds),
        "reward": np.random.binomial(1, 0.5, n_rounds),
        "expected_reward": np.random.random((n_rounds, n_actions)),
        "pi_b": np.random.random((n_rounds, n_actions)),
        "pscore": np.random.random(n_rounds),
        "position": None,
    }

    # Act
    sut._fit_by_gradiant_based_approach(
        bandit_feedback_train=bandit_feedback_train,
        bandit_feedback_test=bandit_feedback_train,
    )

    # Assert
    assert len(sut.train_losses) > 0, "学習時の損失が記録されていること"
    assert (
        len(sut.train_values) > 0
    ), "学習データに対する方策性能の推移が記録されていること"
    assert (
        len(sut.test_values) > 0
    ), "テストデータに対する方策性能の推移が記録されていること"


def test_バンディットフィードバックデータを元にDR推定量で勾配ベースのオフ方策学習ができること():
    # Arrange
    n_rounds = 100
    n_actions = 4
    dim_context_features = 200
    dim_action_features = 150
    dim_two_tower_embedding = 100
    off_policy_objective = "dr"

    sut = PolicyByTwoTowerModel(
        dim_context_features,
        dim_action_features,
        dim_two_tower_embedding,
        off_policy_objective=off_policy_objective,
    )
    bandit_feedback_train = {
        "n_rounds": n_rounds,
        "n_actions": n_actions,
        "context": np.random.random((n_rounds, dim_context_features)),
        "action_context": np.random.random((n_actions, dim_action_features)),
        "action": np.random.randint(0, n_actions, n_rounds),
        "reward": np.random.binomial(1, 0.5, n_rounds),
        "expected_reward": np.random.random((n_rounds, n_actions)),
        "pi_b": np.random.random((n_rounds, n_actions)),
        "pscore": np.random.random(n_rounds),
        "position": None,
    }

    # Act
    sut.fit(
        bandit_feedback_train=bandit_feedback_train,
        bandit_feedback_test=bandit_feedback_train,
    )

    # Assert
    assert len(sut.train_losses) > 0, "学習時の損失が記録されていること"
    assert (
        len(sut.train_values) > 0
    ), "学習データに対する方策性能の推移が記録されていること"
    assert (
        len(sut.test_values) > 0
    ), "テストデータに対する方策性能の推移が記録されていること"


def test_バンディットフィードバックデータを元に回帰ベースのオフ方策学習ができること():
    # Arrange
    n_rounds = 100
    n_actions = 4
    dim_context_features = 200
    dim_action_features = 150
    dim_two_tower_embedding = 100
    off_policy_objective = "regression_based"

    sut = PolicyByTwoTowerModel(
        dim_context_features,
        dim_action_features,
        dim_two_tower_embedding,
        off_policy_objective=off_policy_objective,
    )
    bandit_feedback_train = {
        "n_rounds": n_rounds,
        "n_actions": n_actions,
        "context": np.random.random((n_rounds, dim_context_features)),
        "action_context": np.random.random((n_actions, dim_action_features)),
        "action": np.random.randint(0, n_actions, n_rounds),
        "reward": np.random.binomial(1, 0.5, n_rounds),
        "expected_reward": np.random.random((n_rounds, n_actions)),
        "pi_b": np.random.random((n_rounds, n_actions)),
        "pscore": np.random.random(n_rounds),
        "position": None,
    }

    # Act
    sut.fit(
        bandit_feedback_train=bandit_feedback_train,
        bandit_feedback_test=bandit_feedback_train,
    )

    # Assert
    assert len(sut.train_losses) > 0, "学習時の損失が記録されていること"
    assert all(
        [loss is not np.nan for loss in sut.train_losses]
    ), "学習時の損失がnanでないこと"
    assert (
        len(sut.train_values) > 0
    ), "学習データに対する方策性能の推移が記録されていること"
    assert (
        len(sut.test_values) > 0
    ), "テストデータに対する方策性能の推移が記録されていること"
