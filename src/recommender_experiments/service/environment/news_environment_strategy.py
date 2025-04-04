from datetime import datetime
import numpy as np
from recommender_experiments.service.environment.environment_strategy_interface import (
    EnvironmentStrategyInterface,
)
from recommender_experiments.service.synthetic_bandit_feedback import (
    BanditFeedbackModel,
)
import polars as pl
from pandera.polars import DataFrameSchema


class ItemMetadataSchema:
    content_id: str
    content_type: str
    title: str
    summary: str
    embedding: list[float]
    tags: list[str]
    publisher: str
    published_at: datetime


class UserItemInteractionSchema:
    user_id: str
    content_id: str
    interaction_type: str
    timestamp: datetime
    interacted_in: str


class NewsEnvironmentStrategy(EnvironmentStrategyInterface):
    """
    データの対象期間
    - 2024年5月1日から2024年11月30日までの半年間のデータを使用。
    """

    def __init__(
        self,
        item_metadata_df: pl.DataFrame,
        user_item_interaction_df: pl.DataFrame,
    ):
        self.item_metadata_df = item_metadata_df
        self.user_item_interaction_df = user_item_interaction_df

    @property
    def n_actions(self) -> int:
        return 10

    @property
    def dim_context(self) -> int:
        return 100

    @property
    def expected_reward_strategy_name(self) -> str:
        return "実際のデータなので、期待報酬関数は不明"

    @property
    def item_metadata_df(self) -> pl.DataFrame:
        pass

    def obtain_batch_bandit_feedback(self, n_rounds: int) -> BanditFeedbackModel:
        return BanditFeedbackModel(
            n_rounds=n_rounds,
            n_actions=self.n_actions,
            context=np.random.random((n_rounds, self.dim_context)),
            action_context=np.random.random((self.n_actions, self.dim_context)),
            action=np.random.randint(0, self.n_actions, n_rounds),
            position=None,
            reward=np.random.binomial(1, 0.5, n_rounds),
            expected_reward=None,
            pi_b=None,
            pscore=np.random.random(n_rounds),
        )

    def calc_policy_value(
        self,
        expected_reward: np.ndarray,
        action_dist: np.ndarray,
    ) -> float:
        raise Exception("実際のデータなので、真の方策性能は計算できない")
