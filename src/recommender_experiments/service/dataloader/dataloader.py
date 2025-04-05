import abc
from datetime import datetime

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


class ItemMetadataLoaderInterface(abc.ABC):
    @abc.abstractmethod
    def load_as_df(self) -> DataFrame[ItemMetadataSchema]:
        """
        Returns:
            DataFrame[ItemMetadataSchema]: Item metadata as a DataFrame.
        """
        raise NotImplementedError("load_as_df() must be implemented by subclasses.")


class UserMetadataLoaderInterface(abc.ABC):
    @abc.abstractmethod
    def load_as_df(self) -> DataFrame[UserMetadataSchema]:
        """
        Returns:
            DataFrame[UserMetadataSchema]: User metadata as a DataFrame.
        """
        raise NotImplementedError("load_as_df() must be implemented by subclasses.")


class UserItemInteractionLoaderInterface(abc.ABC):
    @abc.abstractmethod
    def load_as_df(self) -> DataFrame[UserItemInteractionSchema]:
        """
        Returns:
            DataFrame[UserItemInteractionSchema]: User-item interaction data as a DataFrame.
        """
        raise NotImplementedError("load_as_df() must be implemented by subclasses.")
