import numpy as np

from recommender_experiments.service.utils.expected_reward_functions import ContextAwareBinary, ContextFreeBinary


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
    sut = ContextFreeBinary(lower=lower, upper=upper)

    # Act
    expected_reward_function = sut.get_function()
    expected_rewards = expected_reward_function(context, action_context)

    # Assert
    assert expected_rewards.shape == (n_rounds, n_actions)
    assert np.all(expected_rewards >= lower), "全てのアクションの期待報酬関数の値は、lower以上である"
    assert np.all(expected_rewards <= upper), "全てのアクションの期待報酬関数の値は、upper以下である"
    assert expected_rewards[0][0] == upper, "index=0のアクションの期待報酬関数は、upperと一致する"
    assert expected_rewards[0][n_actions - 1] == lower, "index=n_actions-1のアクションの期待報酬関数は、lowerと一致する"


def test_内積が大きいペアほど上限値に近く小さいペアほど下限値に近い期待報酬関数になる() -> None:
    # Arrange
    n_rounds = 2
    n_actions = 3
    context = np.array([[1.0, 0.0], [0.0, 1.0]])
    action_context = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    upper = 0.5
    lower = 0.1
    sut = ContextAwareBinary(lower=lower, upper=upper)

    # Act
    expected_reward_function = sut.get_function()
    expected_rewards = expected_reward_function(context, action_context)

    # Assert
    assert expected_rewards.shape == (n_rounds, n_actions)
    assert np.all(expected_rewards >= lower), "全てのアクションの期待報酬関数の値は、lower以上である"
    assert np.all(expected_rewards <= upper), "全てのアクションの期待報酬関数の値は、upper以下である"
    assert expected_rewards[0][0] == upper, (
        "round=0にて、contextとaction_contextの内積が最大のアクションの期待報酬関数は、upperと一致する"
    )
    assert expected_rewards[0][2] == lower, (
        "round=0にて、contextとaction_contextの内積が最小のアクションの期待報酬関数は、lowerと一致する"
    )
    assert expected_rewards[1][2] == upper, (
        "round=1にて、contextとaction_contextの内積が最大のアクションの期待報酬関数は、upperと一致する"
    )
    assert expected_rewards[1][0] == lower, (
        "round=1にて、contextとaction_contextの内積が最小のアクションの期待報酬関数は、lowerと一致する"
    )


def test_逆順を指定する場合は内積が小さいペアほど上限値に近く大きいペアほど下限値に近い期待報酬関数になる() -> None:
    # Arrange
    n_rounds = 2
    n_actions = 3
    context = np.array([[1.0, 0.0], [0.0, 1.0]])
    action_context = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    upper = 0.5
    lower = 0.1
    sut = ContextAwareBinary(lower=lower, upper=upper, should_reverse=True)

    # Act
    expected_reward_function = sut.get_function()
    expected_rewards = expected_reward_function(context, action_context)

    # Assert
    assert expected_rewards.shape == (n_rounds, n_actions)
    assert np.all(expected_rewards >= lower), "全てのアクションの期待報酬関数の値は、lower以上である"
    assert np.all(expected_rewards <= upper), "全てのアクションの期待報酬関数の値は、upper以下である"
    assert expected_rewards[0][0] == lower, (
        "round=0にて、contextとaction_contextの内積が最大のアクションの期待報酬関数は、lowerと一致する"
    )
    assert expected_rewards[0][2] == upper, (
        "round=0にて、contextとaction_contextの内積が最小のアクションの期待報酬関数は、upperと一致する"
    )
    assert expected_rewards[1][2] == lower, (
        "round=1にて、contextとaction_contextの内積が最大のアクションの期待報酬関数は、lowerと一致する"
    )
    assert expected_rewards[1][0] == upper, (
        "round=1にて、contextとaction_contextの内積が最小のアクションの期待報酬関数は、upperと一致する"
    )
