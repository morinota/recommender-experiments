import numpy as np
from recommender_experiments.service.opl.contexual_bandit_policy import (
    ContextualBanditPolicy,
)
from obp.policy import BernoulliTS

from recommender_experiments.service.synthetic_bandit_feedback import (
    BanditFeedbackModel,
)


def test_ContexualBanditモデルがbandit_feedbackを元に学習されること():
    # Arrange
    n_actions = 4
    dim_context = 10
    sut = ContextualBanditPolicy(n_actions=n_actions, dim_context=dim_context)
    bandit_feedback_train = BanditFeedbackModel(
        n_rounds=100,
        n_actions=n_actions,
        context=np.random.random((100, dim_context)),
        action_context=np.random.random((n_actions, dim_context)),
        action=np.random.randint(0, n_actions, 100),
        position=None,
        reward=np.random.binomial(1, 0.5, 100),
        expected_reward=np.random.random((100, n_actions)),
        pi_b=np.random.random((100, n_actions)),
        pscore=np.random.random(100),
    )

    # Act
    sut.fit(bandit_feedback_train)

    # Assert
    assert sut.policy_name == "ContextualBanditPolicy"


def test_ContexualBanditモデルがあるcontextに対する行動選択確率分布を出力できること():
    # Arrange
    n_actions = 4
    dim_context = 10
    sut = ContextualBanditPolicy(n_actions=n_actions, dim_context=dim_context)
    context = np.random.random((5, dim_context))
    action_context = np.random.random((n_actions, dim_context))

    # Act
    action_dist = sut.predict_proba(context, action_context)

    # Assert
    assert isinstance(action_dist, np.ndarray), "行動選択確率分布はndarrayである"
    assert action_dist.shape == (
        5,
        n_actions,
    ), "行動選択確率分布のshapeは(ラウンド数, アクション数)である"
