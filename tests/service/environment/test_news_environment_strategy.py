import polars as pl

from recommender_experiments.service.dataloader.dataloader import (
    DataLoaderInterface,
    UserItemInteractionLoaderInterface,
    UserMetadataLoaderInterface,
)
from recommender_experiments.service.environment.news_environment_strategy import (
    NewsEnvironmentStrategy,
)
from recommender_experiments.service.synthetic_bandit_feedback import (
    BanditFeedbackModel,
)


class DummyItemMetadataLoader(DataLoaderInterface):
    def load_train_interactions(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "content_id": ["1", "2", "3"],
                "content_type": ["news", "movie", "audio"],
                "title": ["Title 1", "Title 2", "Title 3"],
                "summary": ["Summary 1", "Summary 2", "Summary 3"],
                "embedding": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                "tags": [["tag1", "tag2"], ["tag3"], ["tag4", "tag5"]],
                "publisher": ["Publisher 1", "Publisher 2", "Publisher 3"],
                "published_at": [
                    pl.datetime(2024, 5, 1),
                    pl.datetime(2024, 5, 2),
                    pl.datetime(2024, 5, 3),
                ],
            }
        )


class DummyUserMetadataLoader(UserMetadataLoaderInterface):
    def load_as_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "user_id": ["user1", "user2"],
                "profile": ["profile1", "profile2"],
            }
        )


class DummyUserItemInteractionLoader(UserItemInteractionLoaderInterface):
    def load_as_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "user_id": ["user1", "user2"],
                "content_id": ["1", "2"],
                "interaction_type": ["click", "view"],
                "timestamp": [pl.datetime(2024, 5, 1), pl.datetime(2024, 5, 2)],
                "interacted_in": ["feed", "search"],
            }
        )


def test_初期化時には各種データローダーを元にrawデータが正しく取得される():
    # Act
    sut = NewsEnvironmentStrategy(
        item_metadata_loader=DummyItemMetadataLoader(),
        user_metadata_loader=DummyUserMetadataLoader(),
        user_item_interaction_loader=DummyUserItemInteractionLoader(),
    )

    # Assert
    assert sut.n_actions == 3, "正しいアクション数をプロパティに持つこと"
    assert sut.n_users == 2, "正しいユーザ数をプロパティに持つこと"


def test_実際のbanditフィードバックデータを取得できること():
    # Arrange
    sut = NewsEnvironmentStrategy(
        item_metadata_loader=DummyItemMetadataLoader(),
        user_metadata_loader=DummyUserMetadataLoader(),
        user_item_interaction_loader=DummyUserItemInteractionLoader(),
    )

    # Act
    bandit_feedback = sut.obtain_batch_bandit_feedback(n_rounds=10)

    # Assert
    assert type(bandit_feedback) == BanditFeedbackModel
    assert bandit_feedback.n_rounds == 10
