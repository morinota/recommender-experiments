import numpy as np

from recommender_experiments.service.dataloader.dataloader import MINDDataLoader
from recommender_experiments.service.environment.news_environment_strategy import (
    NewsEnvironmentStrategy,
)


def test_実際のMINDデータを使って初期化できること():
    """
    実際のMINDデータセットを使用してNewsEnvironmentStrategyが初期化でき、
    正しいプロパティ値を持つことを確認する
    """
    # Arrange
    data_dir = "data"
    mind_data_loader = MINDDataLoader(data_dir=data_dir)

    # Act
    sut = NewsEnvironmentStrategy(
        mind_data_loader=mind_data_loader,
    )

    # Assert - プロパティの確認
    assert sut.n_actions > 0, "アクション数（ニュース記事数）が正の値であること"
    assert sut.n_actions == 51282, "MINDデータセットのニュース記事数（51,282件）と一致すること"
    
    assert sut.n_users > 0, "ユーザ数が正の値であること"
    assert sut.n_users == 94057, "MINDデータセットのユーザ数（94,057人）と一致すること"
    
    assert sut.dim_context == 100, "コンテキストの次元数が100であること"
    
    assert sut.expected_reward_strategy_name == "実際のデータなので、期待報酬関数は不明", "期待報酬戦略名が正しいこと"


def test_実際のMINDデータを使ってバンディットフィードバックを取得できること():
    """
    実際のMINDデータセットを使用してバンディットフィードバックが生成でき、
    期待する形状と制約を満たすことを確認する
    """
    # Arrange
    data_dir = "data"
    mind_data_loader = MINDDataLoader(data_dir=data_dir)
    sut = NewsEnvironmentStrategy(
        mind_data_loader=mind_data_loader,
    )
    n_rounds = 100

    # Act
    feedback = sut.obtain_batch_bandit_feedback(n_rounds=n_rounds)

    # Assert - 基本的な属性
    assert feedback.n_rounds == n_rounds, f"指定したラウンド数（{n_rounds}）がフィードバックに反映されること"
    assert feedback.n_actions == sut.n_actions, "フィードバックのアクション数が環境のアクション数と一致すること"
    
    # Assert - 各配列の形状
    assert feedback.context.shape == (n_rounds, sut.dim_context), f"コンテキストの形状が({n_rounds}, {sut.dim_context})であること"
    assert feedback.action_context.shape == (sut.n_actions, sut.dim_context), f"アクションコンテキストの形状が({sut.n_actions}, {sut.dim_context})であること"
    assert feedback.action.shape == (n_rounds,), f"アクションの形状が({n_rounds},)であること"
    assert feedback.reward.shape == (n_rounds,), f"報酬の形状が({n_rounds},)であること"
    assert feedback.pscore.shape == (n_rounds,), f"傾向スコアの形状が({n_rounds},)であること"
    
    # Assert - 値の妥当性
    assert all(0 <= a < sut.n_actions for a in feedback.action), "全てのアクションが有効な範囲内（0以上n_actions未満）にあること"
    assert all(r in [0, 1] for r in feedback.reward), "報酬が0または1のバイナリ値であること"
    assert all(0 <= p <= 1 for p in feedback.pscore), "傾向スコアが0以上1以下の範囲にあること"
    
    # Assert - Noneフィールドの確認（実データなので期待報酬などは不明）
    assert feedback.position is None, "positionがNoneであること（使用しない）"
    assert feedback.expected_reward is None, "expected_rewardがNoneであること（実データなので不明）"
    assert feedback.pi_b is None, "pi_bがNoneであること（実データなので不明）"
    
    # Assert - データ型の確認
    assert feedback.context.dtype == np.float64, "コンテキストがfloat64型であること"
    assert feedback.action_context.dtype == np.float64, "アクションコンテキストがfloat64型であること"
    assert feedback.action.dtype in [np.int32, np.int64], "アクションが整数型であること"
    assert feedback.reward.dtype in [np.int32, np.int64], "報酬が整数型であること"
    assert feedback.pscore.dtype == np.float64, "傾向スコアがfloat64型であること"


def test_異なるラウンド数でバンディットフィードバックを取得できること():
    """
    様々なラウンド数でバンディットフィードバックが正しく生成されることを確認する
    """
    # Arrange
    data_dir = "data"
    mind_data_loader = MINDDataLoader(data_dir=data_dir)
    sut = NewsEnvironmentStrategy(
        mind_data_loader=mind_data_loader,
    )
    test_rounds = [1, 10, 100, 1000]

    for n_rounds in test_rounds:
        # Act
        feedback = sut.obtain_batch_bandit_feedback(n_rounds=n_rounds)

        # Assert
        assert feedback.n_rounds == n_rounds, f"n_rounds={n_rounds}の場合、フィードバックのラウンド数が一致すること"
        assert len(feedback.action) == n_rounds, f"n_rounds={n_rounds}の場合、アクション配列の長さが一致すること"
        assert len(feedback.reward) == n_rounds, f"n_rounds={n_rounds}の場合、報酬配列の長さが一致すること"
        assert feedback.context.shape[0] == n_rounds, f"n_rounds={n_rounds}の場合、コンテキストの行数が一致すること"
