import polars as pl

from recommender_experiments.service.dataloader.dataloader import (
    DataLoaderInterface,
)
from recommender_experiments.service.environment.news_environment_strategy import (
    NewsEnvironmentStrategy,
)


class DummyDataLoader(DataLoaderInterface):
    """TDD用のダミーデータローダー"""

    def load_train_interactions(self) -> pl.DataFrame:
        return pl.DataFrame({})

    def load_test_interactions(self) -> pl.DataFrame:
        return pl.DataFrame({})

    def load_all_interactions(self) -> pl.DataFrame:
        return pl.DataFrame({})

    def load_news_metadata(self) -> pl.DataFrame:
        # 3つのニュースアイテムを返す
        return pl.DataFrame(
            {
                "content_id": ["1", "2", "3"],
                "title": ["Title 1", "Title 2", "Title 3"],
                "category": ["news", "movie", "audio"],
            }
        )

    def load_user_metadata(self) -> pl.DataFrame:
        # 2人のユーザーを返す
        return pl.DataFrame(
            {
                "user_id": ["user1", "user2"],
                "profile": ["profile1", "profile2"],
            }
        )


def test_初期化時にニュースメタデータからアクション数が正しく設定される():
    """TDD: NewsEnvironmentStrategyのn_actionsとn_usersプロパティが正しく動作すること"""
    # Arrange
    data_loader = DummyDataLoader()

    # Act
    sut = NewsEnvironmentStrategy(
        mind_data_loader=data_loader,
    )

    # Assert
    assert sut.n_actions == 3, "ニュースメタデータの件数（3件）がアクション数になること"
    assert sut.n_users == 2, "ユーザメタデータの件数（2件）がユーザ数になること"


def test_バンディットフィードバックを取得できること():
    """TDD: NewsEnvironmentStrategyがバンディットフィードバックを取得できること"""
    # Arrange
    data_loader = DummyDataLoader()
    sut = NewsEnvironmentStrategy(
        mind_data_loader=data_loader,
    )

    # Act
    feedback = sut.obtain_batch_bandit_feedback(n_rounds=5)

    # Assert
    assert feedback.n_rounds == 5, "指定したラウンド数がフィードバックに反映されること"
    assert feedback.n_actions == sut.n_actions, "フィードバックのアクション数が環境のアクション数と一致すること"
    assert feedback.context.shape == (5, sut.dim_context), "コンテキストの形状が正しいこと"
    assert feedback.action_context.shape == (sut.n_actions, sut.dim_context), "アクションコンテキストの形状が正しいこと"
    assert feedback.action.shape == (5,), "アクションの形状が正しいこと"
    assert feedback.reward.shape == (5,), "報酬の形状が正しいこと"
    assert feedback.pscore.shape == (5,), "傾向スコアの形状が正しいこと"
    assert all(0 <= a < sut.n_actions for a in feedback.action), "全てのアクションが有効な範囲内にあること"
