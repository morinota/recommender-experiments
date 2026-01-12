"""MINDデータセットの読み込みモジュール."""

import json
from pathlib import Path

import polars as pl


def load_news_df(news_tsv_path: Path) -> pl.DataFrame:
    """news.tsvを読み込んでpolars DataFrameに変換する.

    Args:
        news_tsv_path: news.tsvファイルのパス

    Returns:
        ニュース記事のDataFrame
        カラム: news_id, category, subcategory, title, abstract, url, title_entities, abstract_entities
    """
    column_names = [
        "news_id",
        "category",
        "subcategory",
        "title",
        "abstract",
        "url",
        "title_entities",
        "abstract_entities",
    ]

    df = pl.read_csv(
        news_tsv_path,
        separator="\t",
        has_header=False,
        new_columns=column_names,
        schema_overrides={
            "news_id": pl.Utf8,
            "category": pl.Utf8,
            "subcategory": pl.Utf8,
            "title": pl.Utf8,
            "abstract": pl.Utf8,
            "url": pl.Utf8,
            "title_entities": pl.Utf8,
            "abstract_entities": pl.Utf8,
        },
        null_values=[""],
        quote_char=None,
        infer_schema_length=0,
    )

    return df


def load_behaviors_df(behaviors_tsv_path: Path) -> pl.DataFrame:
    """behaviors.tsvを読み込んでpolars DataFrameに変換する.

    Args:
        behaviors_tsv_path: behaviors.tsvファイルのパス

    Returns:
        ユーザー行動のDataFrame
        カラム: impression_id, user_id, time, history, impressions
    """
    column_names = ["impression_id", "user_id", "time", "history", "impressions"]

    df = pl.read_csv(
        behaviors_tsv_path,
        separator="\t",
        has_header=False,
        new_columns=column_names,
        schema_overrides={
            "impression_id": pl.Int64,
            "user_id": pl.Utf8,
            "time": pl.Utf8,
            "history": pl.Utf8,
            "impressions": pl.Utf8,
        },
        null_values=[""],
    )

    return df


def parse_entities_json(entities_str: str | None) -> int:
    """エンティティのJSON文字列をパースして、エンティティ数を返す.

    Args:
        entities_str: エンティティのJSON文字列

    Returns:
        エンティティの数
    """
    if entities_str is None or entities_str == "":
        return 0

    try:
        entities = json.loads(entities_str)
        return len(entities)
    except (json.JSONDecodeError, TypeError):
        return 0
