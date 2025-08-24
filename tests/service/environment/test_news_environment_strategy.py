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
    sut = NewsEnvironmentStrategy(mind_data_loader=mind_data_loader)

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

    # Assert
    assert feedback.n_rounds == n_rounds, f"指定したラウンド数（{n_rounds}）がフィードバックに反映されること"
    assert feedback.n_actions == sut.n_actions, "フィードバックのアクション数が環境のアクション数と一致すること"

    assert feedback.context.shape == (n_rounds, sut.dim_context), (
        f"コンテキストの形状が({n_rounds}, {sut.dim_context})であること"
    )
    assert feedback.action_context.shape == (sut.n_actions, sut.dim_context), (
        f"アクションコンテキストの形状が({sut.n_actions}, {sut.dim_context})であること"
    )
    assert feedback.action.shape == (n_rounds,), f"アクションの形状が({n_rounds},)であること"
    assert feedback.reward.shape == (n_rounds,), f"報酬の形状が({n_rounds},)であること"

    ## 値の妥当性
    assert all(r in [0, 1] for r in feedback.reward), "報酬が0または1のバイナリ値であること"

    ## Noneフィールドの確認（実データなので期待報酬などは不明なので）
    assert feedback.position is None, "position(表示位置)がNoneであること（MINDデータセットでは不明）"
    assert feedback.expected_reward is None, "expected_reward(真の期待報酬)がNoneであること（実データなので未知）"
    assert feedback.pi_b is None, "pi_b(データ収集方策の確率分布)がNoneであること（MINDデータセットでは不明）"
    assert feedback.pscore is None, "pscore(傾向スコア)がNoneであること（実データなので未知）"

    ## MINDデータセットには51,282件のニュース記事があるので、その範囲内のIDであるべき
    news_metadata = mind_data_loader.load_news_metadata()
    valid_news_ids = list(range(len(news_metadata)))
    assert all(a in valid_news_ids for a in feedback.action), "全てのアクションが実際のニュースIDの範囲内にあること"

    ## 実際のMINDデータの特性を反映した値であることを確認
    # クリック率が現実的な範囲（5-20%程度）にあること
    click_rate = np.mean(feedback.reward)
    assert 0.05 <= click_rate <= 0.20, (
        f"クリック率がMINDデータセットの現実的な範囲（5-20%）にあること: {click_rate:.2%}"
    )

    # 傾向スコア（pscore）は実データでは未知なのでNoneとして削除
    # assert np.std(feedback.pscore) > 0.01, "傾向スコアに適度な分散があること（推薦方策の多様性を反映）"


def test_is_test_dataパラメータによってtrain_testデータが切り替わること():
    """
    is_test_dataパラメータに応じて、train_interactionsまたはtest_interactionsが
    使用されることを確認する
    """
    # Arrange
    data_dir = "data"
    mind_data_loader = MINDDataLoader(data_dir=data_dir)
    sut = NewsEnvironmentStrategy(mind_data_loader=mind_data_loader)
    n_rounds = 50

    # Act - train dataでバンディットフィードバックを取得
    train_feedback = sut.obtain_batch_bandit_feedback(n_rounds=n_rounds, is_test_data=False)

    # Act - test dataでバンディットフィードバックを取得
    test_feedback = sut.obtain_batch_bandit_feedback(n_rounds=n_rounds, is_test_data=True)

    # Assert - 基本的な形状は同じ
    assert train_feedback.n_rounds == test_feedback.n_rounds == n_rounds
    assert train_feedback.n_actions == test_feedback.n_actions == sut.n_actions

    # Assert - train/testで異なるデータが使用されていること
    # (実際のMINDデータのtrain/test interactionsを使用している場合、データが異なるはず)
    assert not np.array_equal(train_feedback.context, test_feedback.context), (
        "train_dataとtest_dataで異なるcontextが生成されること"
    )
    assert not np.array_equal(train_feedback.action, test_feedback.action), (
        "train_dataとtest_dataで異なるactionが選択されること"
    )
    assert not np.array_equal(train_feedback.reward, test_feedback.reward), (
        "train_dataとtest_dataで異なるrewardが生成されること"
    )

    # Assert - それぞれが実際のMINDデータの特性を持つこと
    train_click_rate = np.mean(train_feedback.reward)
    test_click_rate = np.mean(test_feedback.reward)

    # どちらも現実的なクリック率範囲内（異なるシードなので多少の差は許容）
    assert 0.03 <= train_click_rate <= 0.20, f"trainデータのクリック率が現実的範囲内: {train_click_rate:.2%}"
    assert 0.03 <= test_click_rate <= 0.20, f"testデータのクリック率が現実的範囲内: {test_click_rate:.2%}"
