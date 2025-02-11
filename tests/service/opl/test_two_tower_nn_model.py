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
    sut = PolicyByTwoTowerModel(
        dim_context_features,
        dim_action_features,
        dim_two_tower_embedding,
    )

    # Assert
    assert isinstance(sut.context_tower, torch.nn.Sequential)
    assert isinstance(sut.action_tower, torch.nn.Sequential)

    # Context Tower の出力サイズ
    context_output_layer = sut.context_tower[-1]  # 最後が埋め込み層
    assert isinstance(context_output_layer, torch.nn.Linear)
    assert (
        context_output_layer.out_features == 100
    ), "Context Tower の最終出力サイズが dim_two_tower_embedding に一致すること"

    # Action Tower の出力サイズ
    action_output_layer = sut.action_tower[-1]  # 最後が埋め込み層
    assert isinstance(action_output_layer, torch.nn.Linear)
    assert (
        action_output_layer.out_features == 100
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
        1,
    ), "出力の形状が(ラウンド数、アクション数, 1)である。obpの仕様に合わせて1つ軸を追加してる"
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
        1,
    ), "出力の形状が(ラウンド数、アクション数, 1)である。obpの仕様に合わせて1つ軸を追加してる"
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
    sut.fit_by_gradiant_based_approach(
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
    sut.fit_by_gradiant_based_approach(
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


# def test_データ収集方策のpscoreを渡す場合にバンディットフィードバックデータで学習できること():
#     # Arrange
#     n_rounds = 10
#     n_actions = 4
#     dim_context_features = 200
#     dim_action_features = 150
#     dim_two_tower_embedding = 100
#     sut = TwoTowerNNPolicyLearner(
#         dim_context_features,
#         dim_action_features,
#         dim_two_tower_embedding,
#     )

#     # Act
#     sut.fit(
#         context=np.random.random((n_rounds, dim_context_features)),
#         action=np.random.randint(0, n_actions, n_rounds),
#         reward=np.random.binomial(1, 0.5, n_rounds),
#         action_context=np.random.random((n_actions, dim_action_features)),
#         # データ収集方策のpscoreを(0,1)の範囲でランダムに設定
#         pscore=np.random.random(n_rounds),  # shape: (n_rounds,)
#     )
#     action_dist = sut.predict_proba(
#         context=np.random.random((n_rounds, dim_context)),
#         action_context=np.random.random((n_actions, dim_action_features)),
#     )

#     # Assert
#     assert action_dist.shape == (
#         n_rounds,
#         n_actions,
#         1,
#     ), "出力の形状が(ラウンド数、アクション数, 1)である。obpの仕様に合わせて1つ軸を追加してる"
#     assert not np.any(np.isnan(action_dist)), "各要素がnanではないこと"
#     assert np.allclose(
#         action_dist.sum(axis=1), 1.0
#     ), "各ラウンドごとに、確率の総和が1.0"
#     assert np.all(0 <= action_dist) and np.all(
#         action_dist <= 1
#     ), "各アクションの選択確率が0以上1以下であること"
