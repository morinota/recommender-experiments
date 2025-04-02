import numpy as np
from recommender_experiments.service.environment.ranking_synthetic_dataset import RankingSyntheticBanditDataset


def test_ほげ():
    # Arrange
    n_actions = 5
    len_list = 2
    dim_context = 4
    action_context = np.random.randn(n_actions, dim_context)
    sut = RankingSyntheticBanditDataset(
        n_actions=n_actions, len_list=len_list, dim_context=dim_context, action_context=action_context
    )

    # Act
    bandit_feedaback = sut.obtain_batch_bandit_feedback(n_rounds=1000)

    # Assert
    assert bandit_feedaback["n_rounds"] == 1000
    assert bandit_feedaback["n_actions"] == n_actions
    assert bandit_feedaback["dim_context"] == dim_context
    assert bandit_feedaback["context"].shape == (1000 * len_list, dim_context)
    assert bandit_feedaback["action"].shape == (1000 * len_list,)
    assert bandit_feedaback["position"].shape == (1000 * len_list,)
    assert bandit_feedaback["position_level_reward"].shape == (1000 * len_list,)
    assert bandit_feedaback["expected_reward"].shape == (1000 * len_list, n_actions)
