"""バッチ推論パイプライン.

学習済みモデルを使って、ユーザーごとのニュース推薦を生成する。
"""

import pickle
from pathlib import Path

import polars as pl
from loguru import logger

from recommender_experiments.app.mind_ml_project.utils.config import Config


def load_model(model_path: Path):
    """モデルを読み込む."""
    logger.info(f"モデルを読み込み中: {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def load_features_for_inference() -> pl.DataFrame:
    """推論用の特徴量データを読み込む."""
    logger.info("推論用特徴量を読み込み中...")

    # 学習データの一部をサンプリングして推論用データとする(MVPS)
    training_df = pl.read_parquet(Config.FEATURE_STORE_DIR / "training_data.parquet")
    inference_df = training_df.sample(n=1000, seed=123)

    logger.info(f"推論用データ: {inference_df.shape[0]}件")
    return inference_df


def make_predictions(model, inference_df: pl.DataFrame) -> pl.DataFrame:
    """推論を実行し、推薦スコアを計算する."""
    logger.info("推論を実行中...")

    # カテゴリ変数をone-hot encoding
    inference_df = inference_df.to_dummies(columns=["category", "subcategory"])

    # 特徴量を抽出
    feature_columns = [
        col for col in inference_df.columns if col not in ["impression_id", "user_id", "news_id", "clicked"]
    ]
    X = inference_df.select(feature_columns).to_pandas()

    # 予測
    prediction_proba = model.predict_proba(X)[:, 1]

    # 予測結果をDataFrameに追加
    result_df = inference_df.select(["impression_id", "user_id", "news_id", "clicked"]).with_columns(
        pl.Series("prediction_score", prediction_proba)
    )

    logger.info("推論が完了しました")
    return result_df


def create_user_recommendations(predictions_df: pl.DataFrame, top_k: int = 5) -> pl.DataFrame:
    """ユーザーごとのTOP-K推薦を生成する."""
    logger.info(f"TOP-{top_k}推薦を生成中...")

    # ユーザーごとに予測スコアでソートしてTOP-Kを抽出
    recommendations_df = (
        predictions_df.sort(["user_id", "prediction_score"], descending=[False, True])
        .group_by("user_id")
        .agg(
            [
                pl.col("news_id").head(top_k).alias("recommended_news"),
                pl.col("prediction_score").head(top_k).alias("scores"),
            ]
        )
    )

    logger.info(f"推薦を生成しました: {recommendations_df.shape[0]}ユーザー")
    return recommendations_df


def save_recommendations(recommendations_df: pl.DataFrame, output_path: Path) -> None:
    """推薦結果を保存する."""
    logger.info(f"推薦結果を保存中: {output_path}")
    recommendations_df.write_parquet(output_path)
    logger.info("保存が完了しました")


def main() -> None:
    """メイン処理."""
    logger.info("=== バッチ推論パイプライン開始 ===")

    # 必要なディレクトリを作成
    Config.ensure_dirs()

    # 1. モデルの読み込み
    model_path = Config.MODEL_REGISTRY_DIR / "news_recommendation_model.pkl"
    model = load_model(model_path)

    # 2. 推論用データの読み込み
    inference_df = load_features_for_inference()

    # 3. 推論実行
    predictions_df = make_predictions(model, inference_df)

    # 4. ユーザーごとの推薦生成
    recommendations_df = create_user_recommendations(predictions_df, top_k=5)

    # 5. 推薦結果の保存
    output_path = Config.INFERENCE_OUTPUT_DIR / "user_recommendations.parquet"
    save_recommendations(recommendations_df, output_path)

    # 予測結果も保存
    predictions_path = Config.INFERENCE_OUTPUT_DIR / "predictions.parquet"
    predictions_df.write_parquet(predictions_path)
    logger.info(f"予測結果を保存しました: {predictions_path}")

    logger.info("=== バッチ推論パイプラインが完了しました ===")


if __name__ == "__main__":
    main()
