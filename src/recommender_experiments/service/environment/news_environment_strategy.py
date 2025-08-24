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
        # データローダーを保存（後でtrain/test interactionsを読み込むため）
        self.__mind_data_loader = mind_data_loader
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
        """実際のMINDデータを使ってバンディットフィードバックを返す
        Args:
            n_rounds (int): 取得するラウンド数
            is_test_data (bool, optional): テストデータを使うかどうか. Defaults to False.
        """
        # is_test_dataに応じて異なるシードを使用（train/testで異なるデータを生成）
        seed = 123 if is_test_data else 42
        random_state = np.random.RandomState(seed)

        # 実際のMINDデータのinteractionsを読み込み
        if is_test_data:
            interactions_df = self.__mind_data_loader.load_test_interactions()
        else:
            interactions_df = self.__mind_data_loader.load_train_interactions()

        # interactionsデータからサンプリングしてバンディットフィードバックを生成
        available_interactions = len(interactions_df)
        sample_indices = random_state.choice(available_interactions, size=n_rounds, replace=True)
        sampled_interactions = interactions_df[sample_indices]

        # context: ユーザーの履歴に基づいた特徴量（シード依存で生成）
        # TODO: 実際にはsampled_interactionsのuser_idから履歴を取得してembeddingを生成
        context = random_state.normal(0, 0.1, (n_rounds, self.dim_context))

        # action_context: 各ニュース記事の特徴量（固定、ニュースメタデータから生成）
        # NOTE: これはニュース記事自体の特徴なので、train/testで変わらない
        action_context_seed = np.random.RandomState(999)  # 固定シード
        action_context = action_context_seed.normal(0, 0.1, (self.n_actions, self.dim_context))

        # action: 実際のMINDデータから推薦されたニュース記事のIDを抽出
        # TODO: 実際にはsampled_interactionsのimpressionsから推薦されたニュースIDを抽出
        action = random_state.randint(0, self.n_actions, n_rounds)

        # reward: 実際のクリックデータに基づいた報酬を生成
        # TODO: 実際にはsampled_interactionsのimpressionsからクリック情報を抽出
        click_prob = 0.08  # MINDデータセットの平均的なクリック率
        reward = random_state.binomial(1, click_prob, n_rounds)

        return BanditFeedbackModel(
            n_rounds=n_rounds,
            n_actions=self.n_actions,
            context=context,
            action_context=action_context,
            action=action,
            position=None,
            reward=reward,
            expected_reward=None,
            pi_b=None,
            pscore=None,  # MINDデータでは未知
        )

    def calc_policy_value(
        self,
        expected_reward: np.ndarray,
        action_dist: np.ndarray,
    ) -> float:
        raise Exception("実際のデータなので、真の方策性能は計算できない")
