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
        # ニュースIDからインデックスへのマッピングを作成
        self.__news_id_to_index = {
            news_id: idx for idx, news_id in enumerate(self.__item_metadata_df["content_id"].to_list())
        }

    @property
    def n_actions(self) -> int:
        return self.__item_metadata_df.shape[0]

    @property
    def n_users(self) -> int:
        return self.__user_metadata_df.shape[0]

    @property
    def dim_context(self) -> int:
        # 実際のカテゴリ数 + サブカテゴリ数
        categories = self.__item_metadata_df["category"].unique()
        subcategories = self.__item_metadata_df["subcategory"].unique()
        return len(categories) + len(subcategories)

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

        # context: ユーザーの履歴に基づいた特徴量を生成
        context = self.__create_context_from_user_history(sampled_interactions, random_state)

        # action_context: 各ニュース記事の実際の特徴量（ニュースメタデータから生成）
        # NOTE: これはニュース記事自体の特徴なので、train/testで変わらない
        action_context = self.__create_action_context_from_news_metadata()

        # impressionsから実際のアクションとrewardを抽出
        action, reward = self.__extract_actions_and_rewards_from_impressions(sampled_interactions, random_state)

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

    def __extract_actions_and_rewards_from_impressions(
        self, sampled_interactions: pl.DataFrame, random_state: np.random.RandomState
    ) -> tuple[np.ndarray, np.ndarray]:
        """MINDデータセットのimpressionsからアクションと報酬を抽出する

        Args:
            sampled_interactions: サンプリングされたinteractionデータ
            random_state: ランダム状態

        Returns:
            tuple[np.ndarray, np.ndarray]: (actions, rewards)
        """
        actions = []
        rewards = []

        for impressions_str in sampled_interactions["impressions"].to_list():
            # impressions形式: "N55689-1 N35729-0" (ニュースID-クリックフラグ)
            impression_items = impressions_str.split()

            # 有効なニュースIDが見つかるまでリトライ
            # Why: MINDデータセットでは一部のニュースIDがnews.tsvに存在しないことがある
            # （特にtest_interactionsで顕著）ため、有効なニュースIDを確実に取得する必要がある
            valid_found = False
            for _ in range(len(impression_items)):
                selected_impression = random_state.choice(impression_items)
                news_id, click_flag = selected_impression.rsplit("-", 1)

                # ニュースIDをインデックスに変換
                if news_id in self.__news_id_to_index:
                    action_idx = self.__news_id_to_index[news_id]
                    actions.append(action_idx)
                    rewards.append(int(click_flag))
                    valid_found = True
                    break

            if not valid_found:
                # 全てのニュースIDが未知の場合はランダムに選択
                actions.append(random_state.randint(0, self.n_actions))
                rewards.append(0)

        return np.array(actions), np.array(rewards)

    def __create_action_context_from_news_metadata(self) -> np.ndarray:
        """ニュースメタデータから実際の特徴量を生成する

        Returns:
            np.ndarray: ニュース記事の特徴量 (shape: (n_actions, dim_context))
        """
        # カテゴリとサブカテゴリのワンホットエンコーディングを作成
        categories = self.__item_metadata_df["category"].unique().to_list()
        subcategories = self.__item_metadata_df["subcategory"].unique().to_list()

        # カテゴリ・サブカテゴリのマッピング辞書を作成
        category_to_idx = {cat: idx for idx, cat in enumerate(categories)}
        subcategory_to_idx = {subcat: idx + len(categories) for idx, subcat in enumerate(subcategories)}

        action_context = np.zeros((self.n_actions, self.dim_context))

        for i, row in enumerate(self.__item_metadata_df.iter_rows(named=True)):
            category = row["category"]
            subcategory = row["subcategory"]

            # カテゴリのワンホット
            if category in category_to_idx:
                action_context[i, category_to_idx[category]] = 1.0

            # サブカテゴリのワンホット
            if subcategory in subcategory_to_idx:
                action_context[i, subcategory_to_idx[subcategory]] = 1.0

        return action_context

    def __create_context_from_user_history(
        self, sampled_interactions: pl.DataFrame, random_state: np.random.RandomState
    ) -> np.ndarray:
        """ユーザーの履歴からcontext特徴量を生成する

        Args:
            sampled_interactions: サンプリングされたinteractionデータ
            random_state: ランダム状態（履歴がない場合の補完用）

        Returns:
            np.ndarray: ユーザーのcontext特徴量 (shape: (n_rounds, dim_context))
        """
        n_rounds = len(sampled_interactions)
        context = np.zeros((n_rounds, self.dim_context))

        # action_contextを取得（ニュース記事の特徴量マトリックス）
        action_context = self.__create_action_context_from_news_metadata()

        for i, row in enumerate(sampled_interactions.iter_rows(named=True)):
            history_str = row["history"]

            if history_str and history_str.strip():
                # 履歴からニュースIDを取得
                history_news_ids = history_str.strip().split()

                # 履歴のニュースIDをインデックスに変換
                valid_history_indices = []
                for news_id in history_news_ids:
                    if news_id in self.__news_id_to_index:
                        valid_history_indices.append(self.__news_id_to_index[news_id])

                if valid_history_indices:
                    # 履歴のニュース記事の特徴量を平均してユーザープロファイルを作成
                    history_features = action_context[valid_history_indices]
                    user_profile = np.mean(history_features, axis=0)
                    context[i] = user_profile
                else:
                    # 履歴にあるニュースIDが全て未知の場合はランダム特徴量
                    context[i] = random_state.normal(0, 0.1, self.dim_context)
            else:
                # 履歴が空の場合はランダム特徴量
                context[i] = random_state.normal(0, 0.1, self.dim_context)

        return context

    def calc_policy_value(
        self,
        expected_reward: np.ndarray,
        action_dist: np.ndarray,
    ) -> float:
        raise Exception("実際のデータなので、真の方策性能は計算できない")
