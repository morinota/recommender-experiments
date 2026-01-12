"""特徴グループのバックフィルパイプライン.

MINDデータセットから特徴量を抽出し、ローカルのparquetファイルに保存する。
"""

import polars as pl
from loguru import logger

from recommender_experiments.app.mind_ml_project.utils.config import Config
from recommender_experiments.app.mind_ml_project.utils.data_loader import load_behaviors_df, load_news_df
from recommender_experiments.app.mind_ml_project.utils.feature_functions import (
    add_news_features,
    add_user_history_features,
    create_impression_records,
)


def create_news_feature_group() -> None:
    """ニュース記事の特徴グループを作成する."""
    logger.info("ニュース記事の特徴グループを作成中...")

    # 1. データ読み込み
    news_df = load_news_df(Config.MIND_TRAIN_DIR / "news.tsv")
    logger.info(f"ニュース記事データを読み込みました: {news_df.shape[0]}件")

    # 2. 特徴量生成
    news_feature_df = add_news_features(news_df)

    # 3. 必要なカラムのみ選択
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

    # 4. parquetファイルに保存
    output_path = Config.FEATURE_STORE_DIR / "news_features.parquet"
    news_feature_df.write_parquet(output_path)
    logger.info(f"ニュース特徴量を保存しました: {output_path}")


def create_impression_feature_group() -> None:
    """impression特徴グループを作成する."""
    logger.info("impression特徴グループを作成中...")

    # 1. データ読み込み
    behaviors_df = load_behaviors_df(Config.MIND_TRAIN_DIR / "behaviors.tsv")
    logger.info(f"ユーザー行動データを読み込みました: {behaviors_df.shape[0]}件")

    # 2. impression単位のレコードに変換
    impression_df = create_impression_records(behaviors_df)
    logger.info(f"impressionレコードを作成しました: {impression_df.shape[0]}件")

    # 3. ユーザー履歴特徴量を追加
    impression_feature_df = add_user_history_features(impression_df)

    # 4. 必要なカラムのみ選択
    impression_feature_df = impression_feature_df.select(
        [
            "impression_id",
            "user_id",
            "news_id",
            "clicked",
            "history_length",
        ]
    )

    # 5. parquetファイルに保存
    output_path = Config.FEATURE_STORE_DIR / "impression_features.parquet"
    impression_feature_df.write_parquet(output_path)
    logger.info(f"impression特徴量を保存しました: {output_path}")


def create_training_data() -> None:
    """学習データを作成する.

    news_featuresとimpression_featuresをnews_idでjoinして、学習データを作成する。
    """
    logger.info("学習データを作成中...")

    # 1. 特徴量の読み込み
    news_feature_df = pl.read_parquet(Config.FEATURE_STORE_DIR / "news_features.parquet")
    impression_feature_df = pl.read_parquet(Config.FEATURE_STORE_DIR / "impression_features.parquet")

    # 2. news_idでjoin
    training_df = impression_feature_df.join(news_feature_df, on="news_id", how="left")

    # 3. カテゴリをone-hot encodingする準備(まずはそのまま保存)
    logger.info(f"学習データを作成しました: {training_df.shape[0]}件")

    # 4. parquetファイルに保存
    output_path = Config.FEATURE_STORE_DIR / "training_data.parquet"
    training_df.write_parquet(output_path)
    logger.info(f"学習データを保存しました: {output_path}")


def main() -> None:
    """メイン処理."""
    logger.info("=== 特徴グループのバックフィル処理を開始 ===")

    # 必要なディレクトリを作成
    Config.ensure_dirs()

    # 1. ニュース特徴グループを作成
    create_news_feature_group()

    # 2. impression特徴グループを作成
    create_impression_feature_group()

    # 3. 学習データを作成
    create_training_data()

    logger.info("=== 特徴グループのバックフィル処理が完了しました ===")


if __name__ == "__main__":
    main()
