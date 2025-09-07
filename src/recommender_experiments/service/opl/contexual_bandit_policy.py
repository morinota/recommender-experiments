from typing import Optional

import numpy as np
from obp.policy import LogisticTS

from recommender_experiments.service.opl.policy_strategy_interface import (
    PolicyStrategyInterface,
)
from recommender_experiments.service.synthetic_bandit_feedback import (
    BanditFeedbackDict,
    BanditFeedbackModel,
)


class ContextualBanditPolicy(PolicyStrategyInterface):
    def __init__(
        self,
        n_actions: int,
        dim_context: int,
        temperature: float = 1.0,
    ) -> None:
        self.__model = LogisticTS(
            dim=dim_context,
            n_actions=n_actions,
            len_list=1,
            batch_size=1,
            random_state=None,
        )
        self.__temperature = temperature

    @property
    def policy_name(self) -> str:
        return "ContextualBanditPolicy"

    def fit(
        self,
        bandit_feedback_train: BanditFeedbackDict,
        bandit_feedback_test: Optional[BanditFeedbackDict] = None,
    ) -> None:
        contexts = list(bandit_feedback_train["context"])
        actions = list(bandit_feedback_train["action"])
        rewards = list(bandit_feedback_train["reward"])

        for x, a, r in zip(contexts, actions, rewards):
            self.__model.update_params(
                # contextのshapeは(1, dim_context)にする必要あり
                context=x.reshape(1, -1),
                action=a,
                reward=r,
            )

    def predict_proba(self, context: np.ndarray, action_context: np.ndarray, random_state: int = 0) -> np.ndarray:
        n_rounds = context.shape[0]
        n_actions = action_context.shape[0]
        action_dist = np.zeros((n_rounds, n_actions))

        # obp.policy.LogisticTSクラスの実装の都合上、ミニバッチで推論させてる。
        # (まあ推論時間の遅さは実験なので許容してる。実務で使う時は要検討)
        for i in range(n_rounds):
            x = context[i].reshape(1, -1)
            # パラメータの期待値を使って算出された報酬期待値の一覧を取得
            theta = np.array([model.predict_proba(x) for model in self.__model.model_list]).flatten()
            print(f"theta: {theta}")

            # ソフトマックス関数に通して行動選択確率分布を算出
            action_dist[i] = np.exp(theta / self.__temperature) / np.sum(np.exp(theta / self.__temperature))
        return action_dist
