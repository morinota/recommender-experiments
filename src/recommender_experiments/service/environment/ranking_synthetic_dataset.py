import collections
from dataclasses import dataclass
import itertools
from typing import Callable, TypedDict
from obp.dataset import logistic_reward_function, SyntheticBanditDataset, OpenBanditDataset
import numpy as np


class RankingBanditFeedback(TypedDict):
    n_rounds: int  # ラウンド数
    n_actions: int  # アクション数
    dim_context: int  # 特徴量の次元数
    action_context: np.ndarray  # アクション特徴量 (shape: (n_actions, dim_action_features))
    ranking_candidates: list[list[int]]  # ランキング候補の一覧
    action: np.ndarray  # 実際に選択されたアクション (shape: (n_rounds,))
    position: np.ndarray  # ポジション (shape: (n_rounds,))
    reward: np.ndarray  # ポジションレベルの報酬 (shape: (n_rounds * len_list)
    pi_b: np.ndarray  # データ収集方策 P(a|x) (shape: (n_rounds, n_actions))
    pscore: np.ndarray  # 傾向スコア (shape: (n_rounds,))


@dataclass
class RankingSyntheticBanditDataset:
    dim_context: int
    n_actions: int
    len_list: int
    position_level_reward_type: str = "binary"
    position_level_expected_reward_function: Callable = None
    action_context: np.ndarray = None
    behavior_policy_function: Callable = None
    random_state: int = 12345
    dataset_name: str = "ranking_synthetic_bandit_dataset"

    def obtain_batch_bandit_feedback(self, n_rounds: int) -> RankingBanditFeedback:
        """Obtain batch logged bandit data."""
        np.random.seed(self.random_state)

        # コンテキスト生成 (shape: (n_rounds, dim_context))
        contexts = np.random.normal(loc=0, scale=1, size=(n_rounds, self.dim_context))

        # ランキング候補を生成
        ranking_candidates = list(itertools.permutations(range(self.n_actions), self.len_list))
        print(f"{ranking_candidates=}")

        # 期待報酬関数 (position_level_expected_reward_function) が指定されていない場合は、シグモイド内積モデルを使う
        if self.position_level_expected_reward_function is None:
            expected_reward_ = self._default_position_level_reward(contexts)
        position_level_expected_reward_ = self._calc_position_level_expected_reward(contexts)

    def _calc_position_level_expected_reward(self, contexts: np.ndarray) -> np.ndarray:
        """各contextに対して、各アクションのポジションレベルの期待報酬を計算する.
        Args:
            contexts (np.ndarray): 特徴量ベクトル (n_rounds, dim_context)
        Returns:
            np.ndarray: ポジションレベルの期待報酬 (
        """
        n_rounds = contexts.shape[0]

    def _default_position_level_reward(self, contexts: np.ndarray) -> np.ndarray:
        """デフォルトの期待報酬関数:
        - contexts * action_context の内積を計算しsigmoid関数で0-1に変換したものを、
            ポジションレベルの報酬として返す.
        Args:
            contexts (np.ndarray): 特徴量ベクトル (n_rounds, dim_context)
        Returns:
            np.ndarray: ポジションレベルの報酬 (n_rounds, n
        """
        n_rounds = contexts.shape[0]

        # アクション特徴量が指定されていない場合は、ランダムに生成
        if self.action_context is None:
            self.action_context = np.random.normal(loc=0, scale=1, size=(self.n_actions, self.dim_context))

        # ポジションレベルの報酬を計算
        position_level_reward = np.zeros(n_rounds * self.len_list)
        for i in range(n_rounds):
            for k in range(self.len_list):
                position_level_reward[i * self.len_list + k] = np.dot(contexts[i], self.action_context[k])

        return position_level_reward


if __name__ == "__main__":
    # 行動空間が5P2=20通りの、ランキングバンディットデータセットを生成
    dataset = RankingSyntheticBanditDataset(n_actions=5, len_list=2, dim_context=5)
    bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=1000)
    print(bandit_feedback)

    # real_dataset = OpenBanditDataset(behavior_policy="bts", campaign="all")
    # print(f"{real_dataset.n_rounds=}")
    # print(f"{real_dataset.n_actions=}")
    # print(f"{real_dataset.dim_context=}")
    # print(f"{real_dataset.len_list=}")

    # bandit_feedback = real_dataset.obtain_batch_bandit_feedback()
    # print(f"{bandit_feedback.keys()=}")
    # print(f"{bandit_feedback['context'].shape=}")
    # print(f"{bandit_feedback['context'][0:10]}")
    # print(f"{bandit_feedback['position'][0:10]}")
    # print(f"{bandit_feedback['pscore'][0:10]}")

    # synthetic_dataset = SyntheticBanditDataset(
    #     n_actions=real_dataset.n_actions,
    #     dim_context=real_dataset.dim_context,
    #     reward_function=logistic_reward_function,
    #     random_state=12345,
    # )
    # synthetic_bandit_feedback = synthetic_dataset.obtain_batch_bandit_feedback(
    #     n_rounds=1000
    # )
    # print(f"{synthetic_bandit_feedback.keys()=}")
