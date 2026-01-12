"""デイリー特徴量パイプライン.

新しいユーザー行動データとニュース記事データを読み込み、
特徴量ストアに増分更新する。

Note:
    MINDデータセットは静的なデータセットのため、このパイプラインは
    デモンストレーション目的で実装されています。
    本番環境では、新しいimpressionデータとニュース記事データを
    APIやデータベースから取得して処理します。
"""

import polars as pl
from loguru import logger

from recommender_experiments.app.mind_ml_project.utils.config import Config
from recommender_experiments.app.mind_ml_project.utils.data_loader import (
    load_behaviors_df,
    load_news_df,
)
from recommender_experiments.app.mind_ml_project.utils.feature_functions import (
    add_news_features,
    add_user_history_features,
    create_impression_records,
)


def load_new_news_data() -> pl.DataFrame | None:
    """新しいニュース記事データを読み込む.

    Note:
        本番環境では、前回の実行以降に追加されたニュース記事のみを
        APIやデータベースから取得します。
        この実装では、dev データセットから新しいニュースとして読み込みます。

    Returns:
        新しいニュースデータのDataFrame (なければNone)
    """
    logger.info("新しいニュース記事データを確認中...")

    # dev データセットを「新しいデータ」として扱う
    dev_news_path = Config.MIND_DEV_DIR / "news.tsv"
    if not dev_news_path.exists():
        logger.info("新しいニュースデータが見つかりませんでした")
        return None

    news_df = load_news_df(dev_news_path)

    # 既存の特徴量ストアと比較して、新規ニュースのみを抽出
    existing_news_path = Config.FEATURE_STORE_DIR / "news_features.parquet"
    if existing_news_path.exists():
        existing_news_df = pl.read_parquet(existing_news_path)
        existing_news_ids = set(existing_news_df["news_id"].to_list())

        # 新規ニュースのみをフィルタ
        new_news_df = news_df.filter(~pl.col("news_id").is_in(existing_news_ids))
        logger.info(f"新しいニュース記事: {new_news_df.shape[0]}件")

        if new_news_df.shape[0] == 0:
            return None

        return new_news_df
    else:
        logger.info(f"全ニュース記事を新規として処理: {news_df.shape[0]}件")
        return news_df


def load_new_behavior_data() -> pl.DataFrame | None:
    """新しいユーザー行動データを読み込む.

    Note:
        本番環境では、前回の実行以降に発生したimpressionデータのみを
        APIやデータベースから取得します。
        この実装では、dev データセットから新しい行動として読み込みます。

    Returns:
        新しいユーザー行動データのDataFrame (なければNone)
    """
    logger.info("新しいユーザー行動データを確認中...")

    # dev データセットを「新しいデータ」として扱う
    dev_behaviors_path = Config.MIND_DEV_DIR / "behaviors.tsv"
    if not dev_behaviors_path.exists():
        logger.info("新しい行動データが見つかりませんでした")
        return None

    behaviors_df = load_behaviors_df(dev_behaviors_path)

    # 既存の特徴量ストアと比較して、新規impressionのみを抽出
    existing_impression_path = Config.FEATURE_STORE_DIR / "impression_features.parquet"
    if existing_impression_path.exists():
        existing_impression_df = pl.read_parquet(existing_impression_path)
        existing_impression_ids = set(existing_impression_df["impression_id"].to_list())

        # 新規impressionのみをフィルタ
        new_behaviors_df = behaviors_df.filter(~pl.col("impression_id").is_in(existing_impression_ids))
        logger.info(f"新しいimpression: {new_behaviors_df.shape[0]}件")

        if new_behaviors_df.shape[0] == 0:
            return None

        return new_behaviors_df
    else:
        logger.info(f"全impressionを新規として処理: {behaviors_df.shape[0]}件")
        return behaviors_df


def update_news_feature_group(new_news_df: pl.DataFrame) -> None:
    """ニュース特徴グループに新しいデータを追加する.

    Args:
        new_news_df: 新しいニュースデータのDataFrame
    """
    logger.info("ニュース特徴グループを更新中...")

    # 特徴量生成
    news_feature_df = add_news_features(new_news_df)

    # 必要なカラムのみ選択
    news_feature_df = news_feature_df.select(
        [
            "news_id",
            "category",
            "subcategory",
            "title_entity_count",
            "abstract_entity_count",
            "has_abstract",
        ]
    )

    # 既存データに追加
    output_path = Config.FEATURE_STORE_DIR / "news_features.parquet"
    if output_path.exists():
        existing_df = pl.read_parquet(output_path)
        updated_df = pl.concat([existing_df, news_feature_df])
        updated_df.write_parquet(output_path)
        logger.info(f"ニュース特徴量を更新しました: {updated_df.shape[0]}件 (追加: {news_feature_df.shape[0]}件)")
    else:
        news_feature_df.write_parquet(output_path)
        logger.info(f"ニュース特徴量を新規作成しました: {news_feature_df.shape[0]}件")


def update_impression_feature_group(new_behaviors_df: pl.DataFrame) -> None:
    """impression特徴グループに新しいデータを追加する.

    Args:
        new_behaviors_df: 新しいユーザー行動データのDataFrame
    """
    logger.info("impression特徴グループを更新中...")

    # impression単位のレコードに変換
    impression_df = create_impression_records(new_behaviors_df)
    logger.info(f"新しいimpressionレコード: {impression_df.shape[0]}件")

    # ユーザー履歴特徴量を追加
    impression_feature_df = add_user_history_features(impression_df)

    # 必要なカラムのみ選択
    impression_feature_df = impression_feature_df.select(
        [
            "impression_id",
            "user_id",
            "news_id",
            "clicked",
            "history_length",
        ]
    )

    # 既存データに追加
    output_path = Config.FEATURE_STORE_DIR / "impression_features.parquet"
    if output_path.exists():
        existing_df = pl.read_parquet(output_path)
        updated_df = pl.concat([existing_df, impression_feature_df])
        updated_df.write_parquet(output_path)
        logger.info(
            f"impression特徴量を更新しました: {updated_df.shape[0]}件 (追加: {impression_feature_df.shape[0]}件)"
        )
    else:
        impression_feature_df.write_parquet(output_path)
        logger.info(f"impression特徴量を新規作成しました: {impression_feature_df.shape[0]}件")


def update_training_data() -> None:
    """学習データを再生成する.

    Note:
        新しい特徴量が追加されたため、学習データも再作成します。
    """
    logger.info("学習データを再生成中...")

    # 特徴量の読み込み
    news_feature_df = pl.read_parquet(Config.FEATURE_STORE_DIR / "news_features.parquet")
    impression_feature_df = pl.read_parquet(Config.FEATURE_STORE_DIR / "impression_features.parquet")

    # news_idでjoin
    training_df = impression_feature_df.join(news_feature_df, on="news_id", how="left")

    logger.info(f"学習データを再生成しました: {training_df.shape[0]}件")

    # parquetファイルに保存
    output_path = Config.FEATURE_STORE_DIR / "training_data.parquet"
    training_df.write_parquet(output_path)
    logger.info(f"学習データを保存しました: {output_path}")


def main() -> None:
    """メイン処理."""
    logger.info("=== デイリー特徴量パイプライン開始 ===")

    # 必要なディレクトリを作成
    Config.ensure_dirs()

    update_count = 0

    # 1. 新しいニュース記事データの処理
    new_news_df = load_new_news_data()
    if new_news_df is not None:
        update_news_feature_group(new_news_df)
        update_count += 1

    # 2. 新しいユーザー行動データの処理
    new_behaviors_df = load_new_behavior_data()
    if new_behaviors_df is not None:
        update_impression_feature_group(new_behaviors_df)
        update_count += 1

    # 3. 新しいデータがあった場合のみ学習データを再生成
    if update_count > 0:
        update_training_data()
        logger.info("特徴量ストアを更新しました")
    else:
        logger.info("新しいデータがないため、特徴量ストアは更新されませんでした")

    logger.info("=== デイリー特徴量パイプライン完了 ===")


if __name__ == "__main__":
    main()
