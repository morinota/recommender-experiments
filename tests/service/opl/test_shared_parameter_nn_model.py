from recommender_experiments.service.opl.shared_parameter_nn_model import SharedParameterNNPolicyLearner
import numpy as np
import torch


def test_TwoTowerモデルが正しく初期化されること():
    # Arrange
    dim_context = 3
    dim_action_features = 2
    sut = SharedParameterNNPolicyLearner(dim_context=dim_context + dim_action_features)

    # Assert
    assert isinstance(sut.nn_model, torch.nn.Sequential)

    output_layer = sut.nn_model[-2]  # nn.Sequentialの最後から2番目が出力層
    assert isinstance(output_layer, torch.nn.Linear)
    assert output_layer.out_features == 1, (
        "出力層の次元数が1であること(全アクション間でパラメータを共通化しているはずなので)"
    )

    input_layer = sut.nn_model[0]  # nn.Sequentialの最初が入力層
    assert isinstance(input_layer, torch.nn.Linear)
    assert input_layer.in_features == dim_context + dim_action_features, (
        "入力層の次元数がdim_context + dim_action_featuresであること"
    )


def test_アクション候補の数が動的に変化してもアクション選択の確率分布の推論ができること():
    # Arrange
    n_rounds = 10
    n_actions = 4
    dim_context = 3
    dim_action_features = 2
    sut = SharedParameterNNPolicyLearner(dim_context=dim_context + dim_action_features)

    # Act
    action_dist = sut.predict_proba(
        context=np.random.random((n_rounds, dim_context)),
        action_context=np.random.random((n_actions, dim_action_features)),
    )

    # Assert
    assert action_dist.shape == (10, 4, 1), (
        "出力の形状が(ラウンド数、アクション数, 1)である。obpの仕様に合わせて1つ軸を追加してる"
    )
    assert np.allclose(action_dist.sum(axis=1), 1.0), "各ラウンドごとに、確率の総和が1.0"
    assert np.all(0 <= action_dist) and np.all(action_dist <= 1), "各アクションの選択確率が0以上1以下であること"

    # アクション候補の数が変化しても、同一モデルで推論できること
    action_dist = sut.predict_proba(
        context=np.random.random((n_rounds, dim_context)),
        action_context=np.random.random((n_actions + 2, dim_action_features)),
    )
    assert action_dist.shape == (n_rounds, n_actions + 2, 1), (
        "出力の形状が(ラウンド数、アクション数, 1)である。obpの仕様に合わせて1つ軸を追加してる"
    )
    assert np.allclose(action_dist.sum(axis=1), 1.0), "各ラウンドごとに、確率の総和が1.0"
    assert np.all(0 <= action_dist) and np.all(action_dist <= 1), "各アクションの選択確率が0以上1以下であること"
    assert not np.allclose(action_dist[0], action_dist[1])


def test_データ収集方策のpscoreを渡さない場合にバンディットフィードバックデータで学習できること():
    # Arrange
    n_rounds = 10
    n_actions = 4
    dim_context = 3
    dim_action_features = 2
    sut = SharedParameterNNPolicyLearner(dim_context=dim_context + dim_action_features)

    # Act
    sut.fit(
        context=np.random.random((n_rounds, dim_context)),
        action=np.random.randint(0, n_actions, n_rounds),
        reward=np.random.binomial(1, 0.5, n_rounds),
        action_context=np.random.random((n_actions, dim_action_features)),
    )


def test_データ収集方策のpscoreを渡す場合にバンディットフィードバックデータで学習できること():
    # Arrange
    n_rounds = 10
    n_actions = 4
    dim_context = 3
    dim_action_features = 2
    sut = SharedParameterNNPolicyLearner(dim_context=dim_context + dim_action_features)

    # Act
    sut.fit(
        context=np.random.random((n_rounds, dim_context)),
        action=np.random.randint(0, n_actions, n_rounds),
        reward=np.random.binomial(1, 0.5, n_rounds),
        action_context=np.random.random((n_actions, dim_action_features)),
        # データ収集方策のpscoreを(0,1)の範囲でランダムに設定
        pscore=np.random.random(n_rounds),
    )
