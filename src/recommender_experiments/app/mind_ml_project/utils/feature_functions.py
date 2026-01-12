"""特徴量生成関数モジュール."""

import polars as pl

from recommender_experiments.app.mind_ml_project.utils.data_loader import parse_entities_json


def add_news_features(news_df: pl.DataFrame) -> pl.DataFrame:
    """ニュース記事DataFrameに特徴量を追加する.

    Args:
        news_df: ニュース記事のDataFrame

    Returns:
        特徴量が追加されたDataFrame
        追加カラム: title_entity_count, abstract_entity_count, has_abstract
    """
    result_df = news_df.with_columns(
        [
            # title_entitiesカラムから エンティティ数を計算
            pl.col("title_entities")
            .map_elements(parse_entities_json, return_dtype=pl.Int64)
            .alias("title_entity_count"),
            # abstract_entitiesカラムから エンティティ数を計算
            pl.col("abstract_entities")
            .map_elements(parse_entities_json, return_dtype=pl.Int64)
            .alias("abstract_entity_count"),
            # abstractの有無フラグ
            pl.col("abstract").is_not_null().alias("has_abstract"),
        ]
    )

    return result_df


def create_impression_records(behaviors_df: pl.DataFrame) -> pl.DataFrame:
    """behaviors DataFrameから、impression単位のレコードを作成する.

    Args:
        behaviors_df: ユーザー行動のDataFrame

    Returns:
        impression単位のDataFrame
        カラム: impression_id, user_id, time, history, news_id, clicked
    """
    # impressionsカラムをスペースで分割して、各news_idとclickedを展開
    exploded_df = behaviors_df.with_columns(pl.col("impressions").str.split(" ").alias("impression_list")).explode(
        "impression_list"
    )

    # impression_listカラムを news_id と clicked に分割
    result_df = exploded_df.with_columns(
        [
            pl.col("impression_list").str.split("-").list.get(0).alias("news_id"),
            pl.col("impression_list").str.split("-").list.get(1).cast(pl.Int32).alias("clicked"),
        ]
    ).drop("impression_list", "impressions")

    return result_df


def add_user_history_features(impression_df: pl.DataFrame) -> pl.DataFrame:
    """impression DataFrameにユーザー履歴特徴量を追加する.

    Args:
        impression_df: impression単位のDataFrame

    Returns:
        ユーザー履歴特徴量が追加されたDataFrame
        追加カラム: history_length
    """
    result_df = impression_df.with_columns(
        [
            # historyカラムをスペースで分割して、長さを計算
            pl.when(pl.col("history").is_null())
            .then(0)
            .otherwise(pl.col("history").str.split(" ").list.len())
            .alias("history_length"),
        ]
    )

    return result_df
