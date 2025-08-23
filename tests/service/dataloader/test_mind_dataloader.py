import polars as pl

from recommender_experiments.service.dataloader.dataloader import MINDDataLoader


def test_MINDデータセットのinteractionデータが正常にロードできること():
    # Arrange
    data_dir = "data"
    sut = MINDDataLoader(data_dir=data_dir)

    # Act
    train_interaction_df = sut.load_train_interactions()
    test_interaction_df = sut.load_test_interactions()

    print(train_interaction_df.head())
    print(test_interaction_df.head())

    # Assert
    assert isinstance(train_interaction_df, pl.DataFrame), "train_interactionsがDataFrameであること"
    assert isinstance(test_interaction_df, pl.DataFrame), "test_interactionsがDataFrameであること"


def test_MINDデータセットのnewsメタデータが正常にロードできること():
    # Arrange
    data_dir = "data"
    sut = MINDDataLoader(data_dir=data_dir)

    # Act
    news_metadata_df = sut.load_news_metadata()

    print(news_metadata_df.head())

    # Assert
    assert isinstance(news_metadata_df, pl.DataFrame), "news_metadataがDataFrameであること"
