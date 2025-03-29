import abc
from dataclasses import dataclass

import numpy as np


@dataclass
class PolicyStrategy(abc.ABC):
    """
    意思決定方策が実装する共通のインターフェース(Strategy patternにおけるStrategy)
    """

    @abc.abstractmethod
    def fit(self) -> None:
        """bandit feedbackを受け取って方策を更新するメソッド"""
        raise NotImplementedError

    @abc.abstractmethod
    def predict_proba(self) -> np.ndarray:
        """contextを受け取って、方策による行動選択確率 \pi(a|x) を予測するメソッド
        Args:
            context (np.ndarray): コンテキスト特徴量の配列 (n_rounds, dim_context_features)
            action_context (np.ndarray): アクション特徴量の配列 (n_actions, dim_action_features)
        Returns:
            np.ndarray: 行動選択確率 \pi_{\theta}(a|x) の配列 (n_rounds, n_actions, 1)
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def policy_name(self) -> str:
        """方策の名前を表す文字列"""
        raise NotImplementedError
