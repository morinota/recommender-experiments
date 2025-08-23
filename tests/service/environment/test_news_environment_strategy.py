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
        item_metadata_loader=data_loader,
    )

    # Assert
    assert sut.n_actions == 3, "ニュースメタデータの件数（3件）がアクション数になること"
    assert sut.n_users == 2, "ユーザメタデータの件数（2件）がユーザ数になること"


