import numpy as np
from recommender_experiments.service.environment.synthetic_environment_strategy import (
    SyntheticEnvironmentStrategy,
)
from recommender_experiments.service.opl.policy_strategy_interface import (
    PolicyStrategyInterface,
)
from recommender_experiments.service.synthetic_bandit_feedback import BanditFeedbackDict


class DummyPolicyStrategy(PolicyStrategyInterface):
    def fit(self) -> None:
        pass

    def predict_proba(
        self, context: np.ndarray, action_context: np.ndarray, random_state: int = 0
    ) -> np.ndarray:
        n_rounds = context.shape[0]
        n_actions = action_context.shape[0]
        # dummryなので、contextに依らず全てのroundで一様な行動選択確率を返す!
        return np.full((n_rounds, n_actions), 1.0 / n_actions)

    @property
    def policy_name(self) -> str:
        return "DummyPolicyStrategy"


def test_擬似的な設定に基づいてバンディットフィードバックデータを生成できること():
    # Arrange
    sut = SyntheticEnvironmentStrategy()

    # Act
    bandit_feedback = sut.obtain_batch_bandit_feedback(
        logging_policy_strategy=DummyPolicyStrategy(),
        n_rounds=1000,
    )

    # Assert
    assert isinstance(
        bandit_feedback, dict
    ), "バンディットフィードバックが辞書型であること"
    assert (
        "context" in bandit_feedback
    ), "バンディットフィードバックにcontextが含まれていること"
    assert (
        bandit_feedback["context"].shape[0] == 1000
    ), "コンテキストの数がn_roundsと一致すること"
