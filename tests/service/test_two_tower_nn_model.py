import numpy as np
import torch

from recommender_experiments.service.opl.two_tower_nn_model import (
    TwoTowerNNPolicyLearner,
)


def test_TwoTowerモデルが正しく初期化されること():
    # Arrange
    sut = TwoTowerNNPolicyLearner(
        dim_context=300,
        dim_action_features=200,
        dim_two_tower_embedding=100,
    )

    # Assert
    assert isinstance(sut.context_tower, torch.nn.Sequential)
    assert isinstance(sut.action_tower, torch.nn.Sequential)
    assert isinstance(sut.output_layer, torch.nn.Linear)

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

    # スコア予測層の入出力サイズ
    assert (
        sut.output_layer.in_features == 200
    ), "入力サイズが dim_two_tower_embedding * 2 であること"
    assert sut.output_layer.out_features == 1, "出力サイズが1であること"


def test_アクション候補の数が動的に変化してもアクション選択の確率分布の推論ができること():
    # Arrange
    n_rounds = 10
    n_actions = 4
    dim_context = 30
    dim_action_features = 20
    dim_two_tower_embedding = 10
    sut = TwoTowerNNPolicyLearner(
        dim_context=dim_context,
        dim_action_features=dim_action_features,
        dim_two_tower_embedding=dim_two_tower_embedding,
    )

    # Act
    action_dist = sut.predict_proba(
        context=np.random.random((n_rounds, dim_context)),
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
    ), "各アクションの選択確率が0以上1以下であること"

    # アクション候補の数が変化しても、同一モデルで推論できること
    action_dist = sut.predict_proba(
        context=np.random.random((n_rounds, dim_context)),
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


def test_データ収集方策のpscoreを渡さない場合にバンディットフィードバックデータで学習できること():
    # Arrange
    n_rounds = 10
    n_actions = 4
    dim_context = 300
    dim_action_features = 200
    dim_two_tower_embedding = 100
    sut = TwoTowerNNPolicyLearner(
        dim_context=dim_context,
        dim_action_features=dim_action_features,
        dim_two_tower_embedding=dim_two_tower_embedding,
    )

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
    dim_context = 300
    dim_action_features = 200
    dim_two_tower_embedding = 100
    sut = TwoTowerNNPolicyLearner(
        dim_context=dim_context,
        dim_action_features=dim_action_features,
        dim_two_tower_embedding=dim_two_tower_embedding,
    )

    # Act
    sut.fit(
        context=np.random.random((n_rounds, dim_context)),
        action=np.random.randint(0, n_actions, n_rounds),
        reward=np.random.binomial(1, 0.5, n_rounds),
        action_context=np.random.random((n_actions, dim_action_features)),
        # データ収集方策のpscoreを(0,1)の範囲でランダムに設定
        pscore=np.random.random(n_rounds),  # shape: (n_rounds,)
    )
