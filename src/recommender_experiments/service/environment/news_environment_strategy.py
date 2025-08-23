import numpy as np
import polars as pl

from recommender_experiments.service.dataloader.dataloader import (
    DataLoaderInterface,
)
from recommender_experiments.service.environment.environment_strategy_interface import (
    EnvironmentStrategyInterface,
)
from recommender_experiments.service.synthetic_bandit_feedback import (
    BanditFeedbackModel,
)


class NewsEnvironmentStrategy(EnvironmentStrategyInterface):
    """
    データの対象期間
    - 2024年5月1日から2024年11月30日までの半年間のデータを使用。
    """

    def __init__(
        self,
        mind_data_loader: DataLoaderInterface,
    ):
        # データローダーからニュースメタデータを読み込む
        self.__item_metadata_df = mind_data_loader.load_news_metadata()
        # データローダーからユーザメタデータを読み込む
        self.__user_metadata_df = mind_data_loader.load_user_metadata()

    @property
    def n_actions(self) -> int:
        return self.__item_metadata_df.shape[0]

    @property
    def n_users(self) -> int:
        return self.__user_metadata_df.shape[0]

    @property
    def dim_context(self) -> int:
        return 100

    @property
    def expected_reward_strategy_name(self) -> str:
        return "実際のデータなので、期待報酬関数は不明"

    def obtain_batch_bandit_feedback(
        self,
        n_rounds: int,
        is_test_data: bool = False,
    ) -> BanditFeedbackModel:
        """実際のMINDデータなので、mind_data_loaderから読み込んだデータを使ってバンディットフィードバックを返す
        Args:
            n_rounds (int): 取得するラウンド数
            is_test_data (bool, optional): テストデータを使うかどうか. Defaults to False.
        """
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
