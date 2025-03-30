# 実験用に、真の期待報酬関数 E_{p(r|x,a)}[r] を定義するモジュール


import abc
from typing import Callable
import numpy as np


class ExpectedRewardStrategy(abc.ABC):
    @abc.abstractmethod
    def get_function(self) -> Callable[[np.ndarray, np.ndarray, int], np.ndarray]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """期待報酬関数の名前を表す文字列"""
        raise NotImplementedError


class ContextFreeBinary(ExpectedRewardStrategy):
    """(アクションa, 文脈x)の各組み合わせに対する期待報酬 E_{p(r|x,a)}[r] を定義する関数。
    今回は、文脈xに依存しない、アクション毎に固定のcontext-freeな期待報酬関数を想定している。
    具体的には、アクションaのindexが0から大きくなるにつれて、期待報酬がupperからlowerに線形に減少するような関数を想定している。
    Args:
        context (np.ndarray): 文脈x。 (n_rounds, dim_context)の配列。
        action_context (np.ndarray): アクション特徴量。 (n_actions, dim_action_context)の配列。
        random_state (int, optional): 乱数シード. Defaults to None.
        lower (float, optional): 期待値の下限値. Defaults to 0.0.
        upper (float, optional): 期待値の上限値. Defaults to 1.0.
    Returns:
        np.ndarray: 期待報酬 (n_rounds, n_actions) の配列。
    """

    def __init__(self, lower: float = 0.0, upper: float = 1.0) -> None:
        self.lower = lower
        self.upper = upper

    @property
    def name(self) -> str:
        return f"コンテキスト考慮なし(context-free)({self.lower} ~ {self.upper})"

    def get_function(self) -> Callable[[np.ndarray, np.ndarray, int], np.ndarray]:
        def _expected_reward_function(
            context: np.ndarray,  # shape: (n_rounds, dim_context)
            action_context: np.ndarray,  # shape: (n_actions, dim_action_context)
            random_state: int = None,
        ) -> np.ndarray:  # (n_rounds, n_actions)
            n_rounds = context.shape[0]
            n_actions = action_context.shape[0]

            # アクションaのindexが0から大きくなるにつれて、期待報酬がupperからlowerに線形に減少する配列を生成
            action_rewards = np.linspace(self.upper, self.lower, n_actions)
            print(action_rewards)

            # 各ラウンドに対して同じ期待報酬を繰り返す
            rewards = np.tile(action_rewards, (n_rounds, 1))
            return rewards

        return _expected_reward_function


class ContextAwareBinary(ExpectedRewardStrategy):
    """(アクションa, 文脈x)の各組み合わせに対する期待報酬 E_{p(r|x,a)}[r] を定義する関数。
    今回は、文脈xに依存する、context-awareな期待報酬関数を想定している。
    具体的には、contextとaction_contextの内積をとり、
    各ラウンドで内積が最小となる(x,a)から、最大となる(x,a)までの期待報酬が、lowerからupperに線形に増加するような関数。
    Args:
        context (np.ndarray): 文脈x。 (n_rounds, dim_context)の配列。
        action_context (np.ndarray): アクション特徴量。 (n_actions, dim_action_context)の配列。
        random_state (int, optional): 乱数シード. Defaults to None.
        lower (float, optional): 期待値の下限値. Defaults to 0.0.
        upper (float, optional): 期待値の上限値. Defaults to 1.0.
        should_reverse (bool, optional): 期待報酬が内積が小さいペアほど上限値に近く、大きいペアほど下限値に近いようにするかどうか。 Defaults to False.
    Returns:
        np.ndarray: 期待報酬 (n_rounds, n_actions) の配列。
    """

    def __init__(
        self,
        lower: float = 0.0,
        upper: float = 1.0,
        should_reverse: bool = False,
    ) -> None:
        self.lower = lower
        self.upper = upper
        self.should_reverse = should_reverse

    @property
    def name(self) -> str:
        return f"コンテキスト考慮あり(context-aware)({self.lower} ~ {self.upper})"

    def get_function(self) -> Callable[[np.ndarray, np.ndarray, int], np.ndarray]:
        def _expected_reward_function(
            context: np.ndarray,  # shape: (n_rounds, dim_context)
            action_context: np.ndarray,  # shape: (n_actions, dim_action_context)
            random_state: int = None,
        ) -> np.ndarray:  # (n_rounds, n_actions)
            n_rounds = context.shape[0]
            n_actions = action_context.shape[0]

            # 各ラウンドでの、contextとaction_contextの内積を計算
            dot_products = context @ action_context.T  # shape: (n_rounds, n_actions)

            # 内積を正規化(0-1の範囲にスケーリング)
            min_dot = dot_products.min(axis=1, keepdims=True)
            max_dot = dot_products.max(axis=1, keepdims=True)
            normalized_dot = (dot_products - min_dot) / (max_dot - min_dot)

            # 逆順を指定する場合は、期待報酬が内積が小さいペアほど上限値に近く、大きいペアほど下限値に近いようにする
            if self.should_reverse:
                expected_rewards = (1 - normalized_dot) * (
                    self.upper - self.lower
                ) + self.lower
            else:
                # 正規化された値をlowerからupperの範囲にスケーリング
                expected_rewards = (
                    normalized_dot * (self.upper - self.lower) + self.lower
                )

            assert expected_rewards.shape == (n_rounds, n_actions)
            return expected_rewards

        return _expected_reward_function
