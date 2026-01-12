"""設定管理モジュール."""

from pathlib import Path


class Config:
    """プロジェクト設定を管理するクラス."""

    # プロジェクトルートディレクトリ
    PROJECT_ROOT = Path(__file__).parents[5]

    # データディレクトリ
    DATA_DIR = PROJECT_ROOT / "data"
    MIND_TRAIN_DIR = DATA_DIR / "MINDsmall_train"
    MIND_DEV_DIR = DATA_DIR / "MINDsmall_dev"

    # 特徴量ストア(ローカルファイルで代替)
    FEATURE_STORE_DIR = DATA_DIR / "feature_store" / "mind_project"

    # モデルレジストリ(ローカルファイルで代替)
    MODEL_REGISTRY_DIR = DATA_DIR / "model_registry" / "mind_project"

    # 推論結果保存先
    INFERENCE_OUTPUT_DIR = DATA_DIR / "inference_output" / "mind_project"

    @classmethod
    def ensure_dirs(cls) -> None:
        """必要なディレクトリを作成する."""
        cls.FEATURE_STORE_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODEL_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        cls.INFERENCE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
