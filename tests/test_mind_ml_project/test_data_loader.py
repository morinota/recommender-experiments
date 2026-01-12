"""data_loaderモジュールのテスト."""

from pathlib import Path

import polars as pl
import pytest

from recommender_experiments.app.mind_ml_project.utils.data_loader import (
    load_behaviors_df,
    load_news_df,
    parse_entities_json,
)


def test_load_news_dfで指定されたtsvファイルからニュースデータが読み込めること() -> None:
    # Arrange
    news_tsv_path = Path(__file__).parents[2] / "data" / "MINDsmall_train" / "news.tsv"
    if not news_tsv_path.exists():
        pytest.skip(f"テストデータが存在しません: {news_tsv_path}")

    # Act
    news_df = load_news_df(news_tsv_path)

    # Assert
    assert isinstance(news_df, pl.DataFrame), "DataFrameが返されること"
    assert news_df.shape[0] > 0, "データが読み込まれていること"
    expected_columns = [
        "news_id",
        "category",
        "subcategory",
        "title",
        "abstract",
        "url",
        "title_entities",
        "abstract_entities",
    ]
    assert news_df.columns == expected_columns, f"期待されるカラムが存在すること: {expected_columns}"


def test_load_behaviors_dfで指定されたtsvファイルからユーザー行動データが読み込めること() -> None:
    # Arrange
    behaviors_tsv_path = Path(__file__).parents[2] / "data" / "MINDsmall_train" / "behaviors.tsv"
    if not behaviors_tsv_path.exists():
        pytest.skip(f"テストデータが存在しません: {behaviors_tsv_path}")

    # Act
    behaviors_df = load_behaviors_df(behaviors_tsv_path)

    # Assert
    assert isinstance(behaviors_df, pl.DataFrame), "DataFrameが返されること"
    assert behaviors_df.shape[0] > 0, "データが読み込まれていること"
    expected_columns = ["impression_id", "user_id", "time", "history", "impressions"]
    assert behaviors_df.columns == expected_columns, f"期待されるカラムが存在すること: {expected_columns}"


def test_parse_entities_jsonで正しいJSON文字列からエンティティ数が取得できること() -> None:
    # Arrange
    entities_str = '[{"Label": "Microsoft", "Type": "O"}, {"Label": "Apple", "Type": "O"}]'

    # Act
    entity_count = parse_entities_json(entities_str)

    # Assert
    assert entity_count == 2, "エンティティ数が2であること"


def test_parse_entities_jsonでNone入力時に0が返ること() -> None:
    # Arrange
    entities_str = None

    # Act
    entity_count = parse_entities_json(entities_str)

    # Assert
    assert entity_count == 0, "エンティティ数が0であること"


def test_parse_entities_jsonで空文字列入力時に0が返ること() -> None:
    # Arrange
    entities_str = ""

    # Act
    entity_count = parse_entities_json(entities_str)

    # Assert
    assert entity_count == 0, "エンティティ数が0であること"


def test_parse_entities_jsonで不正なJSON文字列入力時に0が返ること() -> None:
    # Arrange
    entities_str = "invalid json"

    # Act
    entity_count = parse_entities_json(entities_str)

    # Assert
    assert entity_count == 0, "エンティティ数が0であること"
