import abc
from datetime import datetime
from pathlib import Path
from typing import Literal

import polars as pl
from pandera.polars import DataFrameSchema
from pandera.typing.polars import DataFrame


class ItemMetadataSchema(DataFrameSchema):
    content_id: str
    content_type: str
    title: str
    summary: str
    embedding: list[float]
    tags: list[str]
    publisher: str
    published_at: datetime


class UserMetadataSchema(DataFrameSchema):
    user_id: str
    profile: int


class UserItemInteractionSchema(DataFrameSchema):
    user_id: str
    content_id: str
    interaction_type: str
    timestamp: datetime
    interacted_in: str


class DataLoaderInterface(abc.ABC):
    @abc.abstractmethod
    def load_train_interactions(self) -> pl.DataFrame:
        raise NotImplementedError

    @abc.abstractmethod
    def load_test_interactions(self) -> pl.DataFrame:
        raise NotImplementedError

    @abc.abstractmethod
    def load_all_interactions(self) -> pl.DataFrame:
        raise NotImplementedError

    @abc.abstractmethod
    def load_news_metadata(self) -> pl.DataFrame:
        raise NotImplementedError

    @abc.abstractmethod
    def load_user_metadata(self) -> pl.DataFrame:
        raise NotImplementedError


class MINDDataLoader(DataLoaderInterface):
    BEHAVIOR_COLUMNS = ["impression_id", "user_id", "time", "history", "impressions"]
    NEWS_COLUMNS = [
        "content_id",
        "category",
        "subcategory",
        "title",
        "abstract",
        "url",
        "title_entities",
        "abstract_entities",
    ]

    def __init__(
        self,
        data_dir: str | Path = "data",
        mode: Literal["small", "large"] = "small",
    ):
        self.__train_data_dir = Path(data_dir) / "MINDsmall_train"
        self.__test_data_dir = Path(data_dir) / "MINDsmall_dev"

        self.__behavior_filename = "behaviors.tsv"
        self.__news_filename = "news.tsv"
        self.__entity_embedding_filename = "entity_embedding.vec"
        self.__relation_embedding_filename = "relation_embedding.vec"

    def load_train_interactions(self) -> pl.DataFrame:
        return pl.read_csv(
            self.__train_data_dir / self.__behavior_filename,
            separator="\t",
            has_header=False,
            new_columns=self.BEHAVIOR_COLUMNS,
        )

    def load_test_interactions(self) -> pl.DataFrame:
        return pl.read_csv(
            self.__test_data_dir / self.__behavior_filename,
            separator="\t",
            has_header=False,
            new_columns=self.BEHAVIOR_COLUMNS,
        )

    def load_all_interactions(self) -> pl.DataFrame:
        return pl.concat([self.load_train_interactions(), self.load_test_interactions()])

    def load_news_metadata(self) -> pl.DataFrame:
        news_df = pl.read_csv(
            self.__train_data_dir / self.__news_filename,
            separator="\t",
            has_header=False,
            new_columns=self.NEWS_COLUMNS,
            quote_char=None,  # tsvの値の中に"が含まれてるため
        )

        entity_embeddings = self._load_embeddings(self.__train_data_dir / self.__entity_embedding_filename)
        relation_embeddings = self._load_embeddings(self.__train_data_dir / self.__relation_embedding_filename)

        return news_df

    def load_user_metadata(self) -> pl.DataFrame:
        user_ids = self.load_all_interactions()["user_id"].unique().to_list()
        return pl.DataFrame(
            {
                "user_id": user_ids,
                "profile": ["hoge"] * len(user_ids),
            }
        )

    def _load_embeddings(self, filepath: Path) -> pl.DataFrame:
        """
        .vecファイルを読み込んで、idカラムとembeddingカラムを持つDataFrameに変換する
        """
        embeddings_df = pl.read_csv(
            filepath,
            separator="	",
            has_header=False,
        )

        embeddings_df = (
            # 1つ目のカラムはidである
            embeddings_df.rename({"column_1": "id"})
            # 最後のカラムは全てnullになっちゃってるので落とす(TODO:根本対応)
            .drop(embeddings_df.columns[-1])
            # id以外のカラムをlist[float]型のembeddingカラムにまとめる
            .with_columns(pl.concat_list(pl.all().exclude("id")).alias("embedding"))
            .select("id", "embedding")
        )

        return embeddings_df
