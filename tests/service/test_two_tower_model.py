from recommender_experiments.service.opl.two_tower_model import TwoTowerPolicyLearner
import numpy as np
import torch


def test_TwoTowerモデルが正しく初期化されること() -> None:
    # Arrange
    dim_context = 3
    dim_action_features = 2
    policy = TwoTowerPolicyLearner(dim_context=dim_context + dim_action_features)

    # Assert
    assert isinstance(policy.nn_model, torch.nn.Sequential)

    output_layer = policy.nn_model[-2]  # nn.Sequentialの最後から2番目が出力層
    assert isinstance(output_layer, torch.nn.Linear)
    assert (
        output_layer.out_features == 1
    ), "出力層の次元数が1であること(全アクション間でパラメータを共通化しているはずなので)"

    input_layer = policy.nn_model[0]  # nn.Sequentialの最初が入力層
    assert isinstance(input_layer, torch.nn.Linear)
    assert (
        input_layer.in_features == dim_context + dim_action_features
    ), "入力層の次元数がdim_context + dim_action_featuresであること"


def test_アクション選択の確率分布の推論ができること() -> None:
    # Arrange
    n_rounds = 10
    n_actions = 4
    dim_context = 3
    dim_action_features = 2
    context = np.random.random((n_rounds, dim_context))
    action_context = np.random.random((n_actions, dim_action_features))
    policy = TwoTowerPolicyLearner(dim_context=dim_context + dim_action_features)

    # Act
    action_dist = policy.predict_proba(context, action_context)

    # Assert
    assert action_dist.shape == (n_rounds, n_actions)

    # 各ラウンドごとの確率の総和が1.0であること
    assert np.allclose(action_dist.sum(axis=1), 1.0)

    # 各要素が0以上1以下であること
    assert np.all(0 <= action_dist) and np.all(action_dist <= 1)

    # 各ラウンドごとの確率分布が異なること
    assert not np.allclose(action_dist[0], action_dist[1])

    # アクション数が増えても、同一モデルで呼び出せること
    n_actions += 2
    action_context = np.random.random((n_actions, dim_action_features))
    action_dist = policy.predict_proba(context, action_context)
    assert action_dist.shape == (n_rounds, n_actions)
    assert np.allclose(action_dist.sum(axis=1), 1.0)
    assert np.all(0 <= action_dist) and np.all(action_dist <= 1)
    assert not np.allclose(action_dist[0], action_dist[1])
