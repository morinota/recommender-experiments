import numpy as np

from recommender_experiments.service.utils.expected_reward_functions import (
    context_free_binary,
)


def test_context_free_binary() -> None:
    # Arrange
    n_rounds = 3
    n_actions = 5
    dim_context = 300
    dim_action_context = 300
    context = np.random.random((n_rounds, dim_context))
    action_context = np.random.random((n_actions, dim_action_context))
    upper = 0.5
    lower = 0.1

    # Act
    expected_rewards = context_free_binary(
        context,
        action_context,
        lower=lower,
        upper=upper,
    )

    # Assert
    assert expected_rewards.shape == (n_rounds, n_actions, 1)
    assert np.all(
        expected_rewards >= lower
    ), "全てのアクションの期待報酬関数の値は、lower以上である"
    assert np.all(
        expected_rewards <= upper
    ), "全てのアクションの期待報酬関数の値は、upper以下である"
    assert (
        expected_rewards[0][0][0] == upper
    ), "index=0のアクションの期待報酬関数は、upperと一致する"
    assert (
        expected_rewards[0][n_actions - 1][0] == lower
    ), "index=n_actions-1のアクションの期待報酬関数は、lowerと一致する"
