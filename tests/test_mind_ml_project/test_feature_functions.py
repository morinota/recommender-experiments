"""feature_functionsモジュールのテスト."""

import polars as pl

from recommender_experiments.app.mind_ml_project.utils.feature_functions import (
    add_news_features,
    add_user_history_features,
    create_impression_records,
)


def test_add_news_featuresでニュース記事に特徴量が追加されること() -> None:
    # Arrange
    news_df = pl.DataFrame(
        {
            "news_id": ["N1", "N2", "N3"],
            "category": ["sports", "news", "entertainment"],
            "subcategory": ["football", "politics", "movies"],
            "title": ["Title1", "Title2", "Title3"],
            "abstract": ["Abstract1", None, "Abstract3"],
            "url": ["url1", "url2", "url3"],
            "title_entities": ['[{"Label": "A"}]', "[]", '[{"Label": "B"}, {"Label": "C"}]'],
            "abstract_entities": ["[]", "[]", '[{"Label": "D"}]'],
        }
    )

    # Act
    result_df = add_news_features(news_df)

    # Assert
    assert "title_entity_count" in result_df.columns, "title_entity_countカラムが追加されていること"
    assert "abstract_entity_count" in result_df.columns, "abstract_entity_countカラムが追加されていること"
    assert "has_abstract" in result_df.columns, "has_abstractカラムが追加されていること"

    assert result_df["title_entity_count"].to_list() == [1, 0, 2], "title_entity_countが正しく計算されていること"
    assert result_df["abstract_entity_count"].to_list() == [0, 0, 1], "abstract_entity_countが正しく計算されていること"
    assert result_df["has_abstract"].to_list() == [True, False, True], "has_abstractが正しく計算されていること"


def test_create_impression_recordsでimpression単位のレコードが作成されること() -> None:
    # Arrange
    behaviors_df = pl.DataFrame(
        {
            "impression_id": [1, 2],
            "user_id": ["U1", "U2"],
            "time": ["2019-11-11 09:00:00", "2019-11-12 10:00:00"],
            "history": ["N1 N2", "N3"],
            "impressions": ["N10-1 N11-0", "N20-0 N21-1 N22-0"],
        }
    )

    # Act
    result_df = create_impression_records(behaviors_df)

    # Assert
    assert "news_id" in result_df.columns, "news_idカラムが存在すること"
    assert "clicked" in result_df.columns, "clickedカラムが存在すること"
    assert result_df.shape[0] == 5, "impressionが展開されて5レコードになること"

    assert result_df["news_id"].to_list() == ["N10", "N11", "N20", "N21", "N22"], "news_idが正しく抽出されていること"
    assert result_df["clicked"].to_list() == [1, 0, 0, 1, 0], "clickedが正しく抽出されていること"


def test_add_user_history_featuresでユーザー履歴特徴量が追加されること() -> None:
    # Arrange
    impression_df = pl.DataFrame(
        {
            "impression_id": [1, 2, 3],
            "user_id": ["U1", "U2", "U3"],
            "time": ["2019-11-11 09:00:00", "2019-11-12 10:00:00", "2019-11-13 11:00:00"],
            "history": ["N1 N2 N3", "N4", None],
            "news_id": ["N10", "N20", "N30"],
            "clicked": [1, 0, 1],
        }
    )

    # Act
    result_df = add_user_history_features(impression_df)

    # Assert
    assert "history_length" in result_df.columns, "history_lengthカラムが追加されていること"
    assert result_df["history_length"].to_list() == [3, 1, 0], "history_lengthが正しく計算されていること"
