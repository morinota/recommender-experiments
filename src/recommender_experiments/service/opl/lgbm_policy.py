from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import GradientBoostingClassifier

from recommender_experiments.service.estimator import estimate_q_x_a_via_regression
from recommender_experiments.service.opl.policy_strategy_interface import PolicyStrategyInterface
from recommender_experiments.service.synthetic_bandit_feedback import BanditFeedbackDict


@dataclass
class LGBMPolicy(PolicyStrategyInterface):
    """LightGBMを使った推薦方策クラス
    実装参考: https://github.com/usaito/www2024-lope/blob/main/notebooks/learning.py
    """

    dim_context_features: int
    dim_action_features: int
    softmax_temprature: float = 1.0
    len_list: int = 1
    batch_size: int = 32
    learning_rate_init: float = 0.005
    alpha: float = 1e-6
    log_eps: float = 1e-10
    max_iter: int = 200
    off_policy_objective: str = "dr"
    random_state: int = 12345

    def __post_init__(self) -> None:
        """Initialize class."""

        self.train_losses = []
        self.train_values = []
        self.test_values = []

        self.model = GradientBoostingClassifier(
            random_state=1234,
        )

    @property
    def policy_name(self) -> str:
        return "LightGBM-based-policy"

    def fit(
        self,
        bandit_feedback_train: BanditFeedbackDict,
        bandit_feedback_test: Optional[BanditFeedbackDict] = None,
    ) -> None:
        context, action, reward, action_context, pscore, pi_b = (
            bandit_feedback_train["context"],
            bandit_feedback_train["action"],
            bandit_feedback_train["reward"],
            bandit_feedback_train["action_context"],
            bandit_feedback_train["pscore"],
            bandit_feedback_train["pi_b"],
        )
        n_rounds = context.shape[0]
        n_actions = action_context.shape[0]

        # action間でモデルパラメータをshareできる設計にしたいので、context特徴量とaction context特徴量をconcatしてTreeモデルの入力とする。
        selected_action_context = action_context[action]
        X = np.hstack([context, selected_action_context])
        self.model.fit(X, reward)

    def predict_proba(self, context: np.ndarray, action_context: np.ndarray, random_state: int = 0) -> np.ndarray:
        """方策による行動選択確率を予測するメソッド
        Args:
            context (np.ndarray): コンテキスト特徴量の配列 (n_rounds, dim_context_features)
            action_context (np.ndarray): アクション特徴量の配列 (n_actions, dim_action_features)
        Returns:
            np.ndarray: 行動選択確率 \pi_{\theta}(a|x) の配列 (n_rounds, n_actions)
        """
        n_rounds = context.shape[0]
        n_actions = action_context.shape[0]

        estimated_rewards = np.zeros((n_rounds, n_actions))
        for i in range(n_rounds):
            context_i = np.tile(context[i], (n_actions, 1))  # (n_actions, dim_context_features)
            X = np.hstack([context_i, action_context])  # (n_actions, dim_context_features + dim_action_features)
            estimated_rewards[i] = self.model.predict_proba(X)[:, 1]  # ポジティブクラス (1) の確率を使う！

        # 温度パラメータでスケーリングしてからソフトマックス
        logits = estimated_rewards / self.softmax_temprature
        action_probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=1).numpy()

        return action_probs

    def sample(
        self,
        context: np.ndarray,
        action_context: np.ndarray,
        random_state: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        pass
