import numpy as np
from recommender_experiments.service.environment.synthetic_environment_strategy import (
    SyntheticEnvironmentStrategy,
)
from recommender_experiments.service.opl.policy_strategy_interface import (
    PolicyStrategyInterface,
)
from recommender_experiments.service.synthetic_bandit_feedback import (
    BanditFeedbackModel,
)


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
    n_actions = 5
    dim_context = 4
    action_context = np.random.randn(n_actions, dim_context)
    sut = SyntheticEnvironmentStrategy(
        n_actions=n_actions,
        dim_context=dim_context,
        action_context=action_context,
    )

    # Act
    actual = sut.obtain_batch_bandit_feedback(
        logging_policy_strategy=DummyPolicyStrategy(),
        n_rounds=100,
    )

    # Assert
    assert isinstance(
        actual, BanditFeedbackModel
    ), "返り値がBanditFeedbackModel型であること"
    assert actual.context.shape[0] == 100, "コンテキストの数がn_roundsと一致すること"
