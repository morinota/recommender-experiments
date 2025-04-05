import tempfile

import polars as pl

from recommender_experiments.service.feature.feature_interface import SampleFeature


def test_サンプルの特徴量が生成されること():
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Arrange
        sut = SampleFeature(
            output_dir=tmpdirname,
            feature_cols=["feature1", "feature2"],
            user_col="user_id",
            item_col="content_id",
            key_cols=["user_id", "content_id"],
        )

        # Act
        df = sut.fit()

        # Assert
        assert isinstance(df, pl.DataFrame), "特徴量がDataFrameで返されること"
        assert df.shape[0] > 0, "1行以上の特徴量が生成されること"
        assert set(df.columns) == set(sut._key_cols + sut.feature_cols), (
            "返り値のdfのカラムがkey_cols + feature_colsであること"
        )


def test_生成した特徴量を保存できること():
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Arrange
        sut = SampleFeature(
            output_dir=tmpdirname,
            feature_cols=["feature1", "feature2"],
            user_col="user_id",
            item_col="content_id",
            key_cols=["user_id", "content_id"],
        )

        # Act
        sut.save(sut.fit())

        # Assert
        assert sut.output_path.exists(), "指定されたパスに特徴量のファイルが生成されること"
        assert sut.output_path.suffix == ".parquet", "特徴量ファイルの拡張子が.parquetであること"
