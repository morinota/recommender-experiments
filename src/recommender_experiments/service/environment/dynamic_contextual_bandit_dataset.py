"""動的候補プール用Contextual Banditデータセット生成器."""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Set
from pydantic import BaseModel, ConfigDict
from sklearn.utils import check_random_state


@dataclass
class ItemLifecycle:
    """アイテムのライフサイクル情報.

    Attributes
    ----------
    item_id : int
        アイテムID
    start_day : int
        候補プールへの追加日（0から開始）
    duration_days : int
        有効期間（日数）
    """

    item_id: int
    start_day: int
    duration_days: int

    @property
    def end_day(self) -> int:
        """除外日を計算（除外日当日は含まない）."""
        return self.start_day + self.duration_days


@dataclass
class DailyBanditData:
    """1日分のバンディットデータ.

    Attributes
    ----------
    day : int
        日付（0から開始）
    available_actions : Set[int]
        その日に利用可能なアイテムIDのセット
    context_features : np.ndarray
        コンテキスト特徴量 (daily_traffic x dim_context)
    selected_actions : np.ndarray
        選択されたアイテムID (daily_traffic,)
    observed_rewards : np.ndarray
        観測された報酬 (daily_traffic,)
    action_probabilities : np.ndarray
        各アイテムの選択確率 (daily_traffic x max_num_actions)
    """

    day: int
    available_actions: Set[int]
    context_features: np.ndarray
    selected_actions: np.ndarray
    observed_rewards: np.ndarray
    action_probabilities: np.ndarray


class CumulativeRewardSimulationData(BaseModel):
    """累積報酬シミュレーションデータ.

    Attributes
    ----------
    days : int
        シミュレーション日数
    daily_traffic : int
        1日あたりのtraffic数
    daily_data : List[DailyBanditData]
        日別のバンディットデータ
    cumulative_rewards : List[float]
        日別の累積報酬
    total_trials : int
        総trial数
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    days: int
    daily_traffic: int
    daily_data: List[DailyBanditData]
    cumulative_rewards: List[float]
    total_trials: int


class DynamicContextualBanditDataset(BaseModel):
    """動的候補プール用Contextual Banditデータセット生成器.

    このクラスは、アイテムのライフサイクル（追加・除外）と
    1日あたりのtraffic数を考慮したContextual Banditシミュレーションを行います。

    Args:
        dim_context (int): コンテキスト特徴量の次元数
        max_num_actions (int): 最大アイテム数（ID空間のサイズ）
        item_lifecycles (List[ItemLifecycle]): アイテムライフサイクルのリスト
        days (int): シミュレーション日数
        daily_traffic (int): 1日あたりのtraffic数
        reward_noise (float): 報酬に加えるガウシアンノイズの標準偏差
        beta (float): ソフトマックス温度パラメータ（負値で決定論的）
        random_state (int): 再現性確保のための乱数シード
    """

    dim_context: int
    max_num_actions: int
    item_lifecycles: List[ItemLifecycle]
    days: int
    daily_traffic: int
    reward_noise: float = 0.1
    beta: float = 1.0
    random_state: int = 42

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context) -> None:
        """初期化後の検証."""
        # アイテムIDが範囲内かチェック
        for lifecycle in self.item_lifecycles:
            if lifecycle.item_id >= self.max_num_actions:
                raise ValueError(
                    f"item_id {lifecycle.item_id} は max_num_actions {self.max_num_actions} を超えています"
                )

        # 日数の妥当性チェック
        if self.days <= 0:
            raise ValueError(f"days は正の値である必要があります: {self.days}")

        # traffic数の妥当性チェック
        if self.daily_traffic <= 0:
            raise ValueError(f"daily_traffic は正の値である必要があります: {self.daily_traffic}")

    def generate_simulation_data(self) -> CumulativeRewardSimulationData:
        """累積報酬シミュレーション用データを生成する.

        Returns
        -------
        CumulativeRewardSimulationData
            生成されたシミュレーションデータ
        """
        random_ = check_random_state(self.random_state)

        # アイテムの特徴量を事前生成（固定）
        item_features = self._generate_item_features(random_)

        daily_data = []
        cumulative_reward = 0.0
        cumulative_rewards = []

        for day in range(self.days):
            # その日に利用可能なアイテムを取得
            available_actions = self._get_available_actions(day)

            if not available_actions:
                # 利用可能なアイテムがない場合はスキップ
                daily_data.append(
                    DailyBanditData(
                        day=day,
                        available_actions=set(),
                        context_features=np.array([]),
                        selected_actions=np.array([]),
                        observed_rewards=np.array([]),
                        action_probabilities=np.array([]),
                    )
                )
                cumulative_rewards.append(cumulative_reward)
                continue

            # 1日分のデータを生成
            day_data = self._generate_daily_data(day, available_actions, item_features, random_)
            daily_data.append(day_data)

            # 累積報酬を更新
            daily_reward = np.sum(day_data.observed_rewards)
            cumulative_reward += daily_reward
            cumulative_rewards.append(cumulative_reward)

        return CumulativeRewardSimulationData(
            days=self.days,
            daily_traffic=self.daily_traffic,
            daily_data=daily_data,
            cumulative_rewards=cumulative_rewards,
            total_trials=self.days * self.daily_traffic,
        )

    def _generate_item_features(self, random_: np.random.RandomState) -> np.ndarray:
        """アイテム特徴量を生成する.

        Parameters
        ----------
        random_ : np.random.RandomState
            乱数生成器

        Returns
        -------
        np.ndarray
            アイテム特徴量 (max_num_actions x dim_context)
        """
        return random_.normal(size=(self.max_num_actions, self.dim_context))

    def _get_available_actions(self, day: int) -> Set[int]:
        """指定日に利用可能なアイテムIDのセットを取得する.

        Parameters
        ----------
        day : int
            対象日（0から開始）

        Returns
        -------
        Set[int]
            利用可能なアイテムIDのセット
        """
        available_actions = set()
        for lifecycle in self.item_lifecycles:
            if lifecycle.start_day <= day < lifecycle.end_day:
                available_actions.add(lifecycle.item_id)
        return available_actions

    def _generate_daily_data(
        self, day: int, available_actions: Set[int], item_features: np.ndarray, random_: np.random.RandomState
    ) -> DailyBanditData:
        """1日分のバンディットデータを生成する.

        Parameters
        ----------
        day : int
            対象日
        available_actions : Set[int]
            利用可能なアイテムIDのセット
        item_features : np.ndarray
            アイテム特徴量
        random_ : np.random.RandomState
            乱数生成器

        Returns
        -------
        DailyBanditData
            1日分のバンディットデータ
        """
        # コンテキスト特徴量を生成
        context_features = random_.normal(size=(self.daily_traffic, self.dim_context))

        # 各trial用の行動選択確率と選択された行動を計算
        selected_actions = []
        all_action_probabilities = []
        observed_rewards = []

        available_action_list = sorted(list(available_actions))

        for i in range(self.daily_traffic):
            context = context_features[i]

            # 利用可能なアイテムに対してのみ報酬期待値を計算
            expected_rewards = self._compute_expected_rewards(context, available_action_list, item_features)

            # ソフトマックス方策で行動選択確率を計算
            action_probs = self._compute_action_probabilities(expected_rewards, available_action_list)
            all_action_probabilities.append(action_probs)

            # 行動をサンプリング
            selected_action = self._sample_action(available_action_list, expected_rewards, random_)
            selected_actions.append(selected_action)

            # 報酬を生成
            reward = self._generate_reward(context, selected_action, item_features, random_)
            observed_rewards.append(reward)

        return DailyBanditData(
            day=day,
            available_actions=available_actions,
            context_features=context_features,
            selected_actions=np.array(selected_actions),
            observed_rewards=np.array(observed_rewards),
            action_probabilities=np.array(all_action_probabilities),
        )

    def _compute_expected_rewards(
        self, context: np.ndarray, available_actions: List[int], item_features: np.ndarray
    ) -> np.ndarray:
        """期待報酬を計算する.

        Parameters
        ----------
        context : np.ndarray
            コンテキスト特徴量 (dim_context,)
        available_actions : List[int]
            利用可能なアイテムIDのリスト
        item_features : np.ndarray
            アイテム特徴量 (max_num_actions x dim_context)

        Returns
        -------
        np.ndarray
            期待報酬 (len(available_actions),)
        """
        expected_rewards = []
        for action_id in available_actions:
            # コンテキストとアイテム特徴量の内積で期待報酬を計算
            reward = np.dot(context, item_features[action_id])
            expected_rewards.append(reward)
        return np.array(expected_rewards)

    def _compute_action_probabilities(self, expected_rewards: np.ndarray, available_actions: List[int]) -> np.ndarray:
        """行動選択確率を計算する（max_num_actionsサイズに拡張）.

        Parameters
        ----------
        expected_rewards : np.ndarray
            期待報酬
        available_actions : List[int]
            利用可能なアイテムIDのリスト

        Returns
        -------
        np.ndarray
            行動選択確率 (max_num_actions,)
        """
        # ソフトマックス関数で確率計算
        exp_rewards = np.exp(self.beta * expected_rewards)
        probs_available = exp_rewards / np.sum(exp_rewards)

        # max_num_actionsサイズに拡張
        full_probs = np.zeros(self.max_num_actions)
        for i, action_id in enumerate(available_actions):
            full_probs[action_id] = probs_available[i]

        return full_probs

    def _sample_action(
        self, available_actions: List[int], expected_rewards: np.ndarray, random_: np.random.RandomState
    ) -> int:
        """行動をサンプリングする.

        Parameters
        ----------
        available_actions : List[int]
            利用可能なアイテムIDのリスト
        expected_rewards : np.ndarray
            期待報酬
        random_ : np.random.RandomState
            乱数生成器

        Returns
        -------
        int
            選択されたアイテムID
        """
        # ソフトマックス確率で行動選択
        exp_rewards = np.exp(self.beta * expected_rewards)
        probs = exp_rewards / np.sum(exp_rewards)

        selected_idx = random_.choice(len(available_actions), p=probs)
        return available_actions[selected_idx]

    def _generate_reward(
        self, context: np.ndarray, action_id: int, item_features: np.ndarray, random_: np.random.RandomState
    ) -> float:
        """報酬を生成する.

        Parameters
        ----------
        context : np.ndarray
            コンテキスト特徴量
        action_id : int
            選択されたアイテムID
        item_features : np.ndarray
            アイテム特徴量
        random_ : np.random.RandomState
            乱数生成器

        Returns
        -------
        float
            観測された報酬
        """
        # 期待報酬を計算
        expected_reward = np.dot(context, item_features[action_id])

        # ノイズを追加
        noise = random_.normal(0, self.reward_noise)

        return expected_reward + noise
