import numpy as np
import torch
from sklearn.ensemble import GradientBoostingClassifier

from recommender_experiments.service.opl.lgbm_policy import LGBMPolicy


def test_LightGBMに基づく推薦方策が正しく初期化されること():
    # Arrange
    dim_context_features = 200
    dim_action_features = 150

    # Act
    sut = LGBMPolicy(dim_context_features, dim_action_features)

    # Assert
    assert sut.policy_name == "LightGBM-based-policy"
    assert isinstance(sut.model, GradientBoostingClassifier), "GradientBoostingClassifierのインスタンスであること"
    assert sut.train_losses == []
    assert sut.train_values == []
    assert sut.test_values == []


def test_banditfeedbackを元に推薦方策の学習ができること():
    # Arrange
    n_rounds = 10
    n_actions = 4
    dim_context_features = 200
    dim_action_features = 150
    sut = LGBMPolicy(dim_context_features, dim_action_features)
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
    sut.fit(bandit_feedback_train)  # Tree系モデルは最初にfitしないと何を推論していいかわからない。
    action_dist = sut.predict_proba(
        context=np.random.random((n_rounds, dim_context_features)),
        action_context=np.random.random((n_actions, dim_action_features)),
    )

    # Assert
    print(f"{action_dist=}")
    assert action_dist.shape == (n_rounds, n_actions), "出力の形状が(ラウンド数、アクション数)であること"
    assert np.allclose(action_dist.sum(axis=1), 1.0), "各ラウンドごとに、確率の総和が1.0"
    assert np.all(0 <= action_dist) and np.all(action_dist <= 1), "各アクションの選択確率が0.0以上1.0以下であること"


def test_アクション候補の数が動的に変化してもアクション選択の確率分布の推論ができること():
    # Arrange
    n_rounds = 10
    n_actions = 4
    dim_context_features = 200
    dim_action_features = 150
    sut = LGBMPolicy(dim_context_features, dim_action_features)
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
    sut.fit(bandit_feedback_train)

    # Act
    action_dist_1 = sut.predict_proba(
        context=np.random.random((n_rounds, dim_context_features)),
        action_context=np.random.random((n_actions, dim_action_features)),
    )
    # アクション候補の数が変化しても、同一モデルで推論できること
    action_dist_2 = sut.predict_proba(
        context=np.random.random((n_rounds, dim_context_features)),
        action_context=np.random.random((n_actions + 2, dim_action_features)),
    )

    # Assert
    assert action_dist_1.shape == (n_rounds, n_actions), "出力の形状が(ラウンド数、アクション数)であること"
    assert np.allclose(action_dist_1.sum(axis=1), 1.0), "各ラウンドごとに、確率の総和が1.0"
    assert np.all(0 <= action_dist_1) and np.all(action_dist_1 <= 1), "各アクションの選択確率が0.0以上1.0以下であること"
    assert action_dist_2.shape == (n_rounds, n_actions + 2), "出力の形状が(ラウンド数、アクション数)であること"
    assert np.allclose(action_dist_2.sum(axis=1), 1.0), "各ラウンドごとに、確率の総和が1.0"
    assert np.all(0 <= action_dist_2) and np.all(action_dist_2 <= 1), "各アクションの選択確率が0以上1以下であること"
