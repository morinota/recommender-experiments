import abc
from pathlib import Path
from typing import Literal
import polars as pl


class Feature(abc.ABC):
    """
    特徴量が実装する共通のabstract baseクラス (Template method patternにおけるTemplate)
    """

    def __init__(
        self,
        output_dir: str | Path,
        # 生成する特徴量の名前たち
        feature_cols: list[str],
        user_col: str = "user_id",
        item_col: str = "content_id",
        # MEMO: 生成した特徴量をbandit feedbackのcontextに紐づけるためのキー
        # 基本的には3択くらい?(user_idのみ, content_idのみ, user_id+content_id)
        # もしくは3択+timestampとか!
        key_cols: list[str] = ["user_id", "content_id"],
        suffix: str | None = None,
        mode: Literal["train", "test"] | None = None,
    ) -> None:
        self.class_name = self.__class__.__name__
        if suffix is not None:
            self.class_name = f"{self.class_name}_{suffix}"
        if mode is not None:
            self.class_name = f"{self.class_name}_{mode}"

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.output_dir / f"{self.class_name}.parquet"
        self.feature_cols = feature_cols

        self._user_col = user_col
        self._item_col = item_col
        self._key_cols = key_cols
        self._suffix = suffix
        self._mode = mode

    @abc.abstractmethod
    def fit(self) -> pl.DataFrame:
        raise NotImplementedError

    def transform(self) -> pl.DataFrame:
        return self.fit()

    def save(self, df: pl.DataFrame) -> None:
        self.validate_feature(df)
        df.write_parquet(self.output_path)

    def load(self) -> pl.DataFrame:
        return pl.read_parquet(self.output_path)

    def validate_feature(self, df: pl.DataFrame) -> None:
        if df.shape[0] == 0:
            raise ValueError("Empty dataframe")


class SampleFeature(Feature):
    def __init__(
        self,
        output_dir: str | Path,
        feature_cols: list[str],
        user_col: str = "user_id",
        item_col: str = "content_id",
        key_cols: list[str] = ["user_id", "content_id"],
        suffix: str | None = None,
    ):
        super().__init__(output_dir, feature_cols, user_col, item_col, key_cols, suffix)
        self.df = pl.DataFrame(
            {
                self._user_col: [1, 1, 1, 2, 2, 2, 3, 3, 3],
                self._item_col: [1, 2, 3, 1, 2, 3, 1, 2, 3],
                "score": [0.1, 0.2, 0.3, 0.2, 0.4, 0.6, 0.3, 0.6, 0.9],
                "rank": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            }
        )

    def fit(self) -> pl.DataFrame:
        df = self.df.with_columns(pl.lit(1).alias("feature1"))
        df = df.with_columns(pl.lit(2).alias("feature2"))
        return df[self._key_cols + self.feature_cols]
