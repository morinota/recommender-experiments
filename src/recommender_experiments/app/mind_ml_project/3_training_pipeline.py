"""トレーニングパイプライン.

学習データを使ってXGBoostモデルを訓練し、モデルレジストリに保存する。
"""

import pickle
from pathlib import Path

import polars as pl
from loguru import logger
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from recommender_experiments.app.mind_ml_project.utils.config import Config


def load_training_data(sample_size: int | None = None) -> pl.DataFrame:
    """学習データを読み込む.

    Args:
        sample_size: サンプリングするデータ数(Noneの場合は全データ)

    Returns:
        学習データのDataFrame
    """
    logger.info("学習データを読み込み中...")
    training_df = pl.read_parquet(Config.FEATURE_STORE_DIR / "training_data.parquet")
    logger.info(f"学習データを読み込みました: {training_df.shape[0]}件")

    if sample_size is not None and sample_size < training_df.shape[0]:
        logger.info(f"データをサンプリングします: {sample_size}件")
        training_df = training_df.sample(n=sample_size, seed=42)

    return training_df


def prepare_features_and_target(training_df: pl.DataFrame) -> tuple:
    """特徴量とターゲットを準備する.

    Args:
        training_df: 学習データのDataFrame

    Returns:
        (X, y)のタプル
        - X: 特徴量のDataFrame (pandas)
        - y: ターゲット (pandas Series)
    """
    logger.info("特徴量とターゲットを準備中...")

    # カテゴリ変数をone-hot encoding
    training_df = training_df.to_dummies(columns=["category", "subcategory"])

    # 特徴量とターゲットに分割
    feature_columns = [
        col for col in training_df.columns if col not in ["impression_id", "user_id", "news_id", "clicked"]
    ]
    X = training_df.select(feature_columns).to_pandas()
    y = training_df.select("clicked").to_pandas()["clicked"]

    logger.info(f"特徴量数: {X.shape[1]}, サンプル数: {X.shape[0]}")
    return X, y


def train_model(X_train, y_train, X_val, y_val) -> GradientBoostingClassifier:
    """GradientBoostingモデルを訓練する.

    Args:
        X_train: 訓練データの特徴量
        y_train: 訓練データのターゲット
        X_val: 検証データの特徴量
        y_val: 検証データのターゲット

    Returns:
        訓練済みGradientBoostingモデル
    """
    logger.info("GradientBoostingモデルの訓練を開始...")

    model = GradientBoostingClassifier(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        verbose=1,
    )

    model.fit(X_train, y_train)

    logger.info("モデルの訓練が完了しました")
    return model


def evaluate_model(model: GradientBoostingClassifier, X_val, y_val) -> dict:
    """モデルを評価する.

    Args:
        model: 訓練済みモデル
        X_val: 検証データの特徴量
        y_val: 検証データのターゲット

    Returns:
        評価メトリクスの辞書
    """
    logger.info("モデルの評価を実施中...")

    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "roc_auc": roc_auc_score(y_val, y_pred_proba),
        "log_loss": log_loss(y_val, y_pred_proba),
    }

    logger.info(f"評価メトリクス: {metrics}")
    return metrics


def save_model(model: GradientBoostingClassifier, model_path: Path) -> None:
    """モデルをpickleファイルに保存する.

    Args:
        model: 訓練済みモデル
        model_path: 保存先パス
    """
    logger.info(f"モデルを保存中: {model_path}")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("モデルの保存が完了しました")


def main() -> None:
    """メイン処理."""
    logger.info("=== トレーニングパイプライン開始 ===")

    # 必要なディレクトリを作成
    Config.ensure_dirs()

    # 1. 学習データの読み込み(MVPSなので10万件でサンプリング)
    training_df = load_training_data(sample_size=100_000)

    # 2. 特徴量とターゲットの準備
    X, y = prepare_features_and_target(training_df)

    # 3. train/validationに分割
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    logger.info(f"訓練データ: {X_train.shape[0]}件, 検証データ: {X_val.shape[0]}件")

    # 4. モデルの訓練
    model = train_model(X_train, y_train, X_val, y_val)

    # 5. モデルの評価
    metrics = evaluate_model(model, X_val, y_val)

    # 6. モデルの保存
    model_path = Config.MODEL_REGISTRY_DIR / "news_recommendation_model.pkl"
    save_model(model, model_path)

    logger.info("=== トレーニングパイプラインが完了しました ===")


if __name__ == "__main__":
    main()
