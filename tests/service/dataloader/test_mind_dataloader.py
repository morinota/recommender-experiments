import polars as pl

from recommender_experiments.service.dataloader.dataloader import MINDDataLoader


def test_MINDデータセットのinteractionデータが正常にロードできること():
    """
    MINDデータセットのユーザー行動データ（behaviors.tsv）が正しく読み込まれ、
    期待するスキーマと形式になっていることを確認する
    """
    # Arrange
    data_dir = "data"
    sut = MINDDataLoader(data_dir=data_dir)

    # Act
    train_interaction_df = sut.load_train_interactions()
    test_interaction_df = sut.load_test_interactions()

    # Assert
    assert isinstance(train_interaction_df, pl.DataFrame), "train_interactionsがDataFrameであること"
    assert isinstance(test_interaction_df, pl.DataFrame), "test_interactionsがDataFrameであること"

    assert len(train_interaction_df) > 0, "学習用インタラクションデータが空でないこと"
    assert len(test_interaction_df) > 0, "テスト用インタラクションデータが空でないこと"

    expected_columns = ["impression_id", "user_id", "time", "history", "impressions"]
    assert train_interaction_df.columns == expected_columns, f"学習データが期待するカラムを持つこと: {expected_columns}"
    assert test_interaction_df.columns == expected_columns, (
        f"テストデータが期待するカラムを持つこと: {expected_columns}"
    )

    # historyカラムの形式確認（スペース区切りの文字列）
    sample_history = train_interaction_df["history"].head(1)[0]
    assert isinstance(sample_history, str), "historyが文字列であること"
    if sample_history:  # 空でない場合
        assert " " in sample_history or len(sample_history.split()) == 1, (
            "historyがスペース区切りまたは単一のニュースIDであること"
        )

    # impressionsカラムの形式確認（スペース区切りのクリック情報付きニュースID）
    sample_impressions = train_interaction_df["impressions"].head(1)[0]
    assert isinstance(sample_impressions, str), "impressionsが文字列であること"
    assert "-" in sample_impressions, "impressionsにクリック情報（-0または-1）が含まれること"


def test_MINDデータセットのnewsメタデータが正常にロードできること():
    """
    MINDデータセットのニュースメタデータ（news.tsv）が正しく読み込まれ、
    期待するスキーマと形式になっていることを確認する
    """
    # Arrange
    data_dir = "data"
    sut = MINDDataLoader(data_dir=data_dir)

    # Act
    news_metadata_df = sut.load_news_metadata()

    # Assert
    assert isinstance(news_metadata_df, pl.DataFrame), "news_metadataがDataFrameであること"
    assert len(news_metadata_df) > 0, "ニュースメタデータが空でないこと"

    expected_columns = [
        "content_id",
        "category",
        "subcategory",
        "title",
        "abstract",
        "url",
        "title_entities",
        "abstract_entities",
    ]
    assert news_metadata_df.columns == expected_columns, f"期待するカラムを持つこと: {expected_columns}"

    # URLの形式確認
    sample_urls = news_metadata_df["url"].head(5)
    for url in sample_urls:
        if url:  # nullでない場合
            assert url.startswith("http"), f"URLがhttpで始まること: {url[:50]}..."

    # カテゴリの多様性確認
    categories = news_metadata_df["category"].unique().to_list()
    assert len(categories) > 0, "少なくとも1つのカテゴリが存在すること"


def test_MINDデータセットのユーザメタデータが正常にロードできること():
    """
    MINDデータセットから生成されるユーザメタデータが正しく読み込まれることを確認する
    """
    # Arrange
    data_dir = "data"
    sut = MINDDataLoader(data_dir=data_dir)

    # Act
    user_metadata_df = sut.load_user_metadata()

    # Assert
    assert isinstance(user_metadata_df, pl.DataFrame), "user_metadataがDataFrameであること"
    assert len(user_metadata_df) > 0, "ユーザメタデータが空でないこと"

    assert "user_id" in user_metadata_df.columns, "user_idカラムが存在すること"
    assert "profile" in user_metadata_df.columns, "profileカラムが存在すること"

    assert user_metadata_df["user_id"].n_unique() == len(user_metadata_df), "user_idが一意であること"

    # user_idの形式（MINDデータセットの形式: U+数字）
    sample_user_ids = user_metadata_df["user_id"].head(10).to_list()
    for user_id in sample_user_ids:
        assert user_id.startswith("U"), f"user_idが'U'で始まること: {user_id}"
        assert user_id[1:].isdigit(), f"user_idの'U'以降が数字であること: {user_id}"


def test_全データ統合が正常に動作すること():
    """
    train/testデータを統合するload_all_interactionsメソッドが正しく動作することを確認する
    """
    # Arrange
    data_dir = "data"
    sut = MINDDataLoader(data_dir=data_dir)

    # Act
    train_df = sut.load_train_interactions()
    test_df = sut.load_test_interactions()
    all_df = sut.load_all_interactions()

    # Assert
    expected_total_rows = len(train_df) + len(test_df)
    assert len(all_df) == expected_total_rows, (
        f"統合データの行数が学習+テストの合計と一致すること: {len(all_df)} == {expected_total_rows}"
    )

    assert all_df.columns == train_df.columns, "統合データのカラムが元データと同じであること"

    for col in all_df.columns:
        assert all_df[col].dtype == train_df[col].dtype, f"{col}のデータ型が保持されていること"
