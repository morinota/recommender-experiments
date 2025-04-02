from typing import Optional
from obp.policy import BernoulliTS
import numpy as np
from recommender_experiments.service.opl.policy_strategy_interface import PolicyStrategyInterface
from recommender_experiments.service.synthetic_bandit_feedback import BanditFeedbackDict


class ContextualBanditPolicy(PolicyStrategyInterface):
    def __init__(self):
        pass

    @property
    def policy_name(self) -> str:
        return "ContextualBanditPolicy"

    def fit(
        self, bandit_feedback_train: BanditFeedbackDict, bandit_feedback_test: Optional[BanditFeedbackDict] = None
    ) -> None:
        pass

    def predict_proba(self, context: np.ndarray, action_context: np.ndarray, random_state: int = 0) -> np.ndarray:
        return np.zeros(action_context.shape[0])
