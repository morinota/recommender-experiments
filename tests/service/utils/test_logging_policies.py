import numpy as np

from recommender_experiments.service.utils.logging_policies import random_policy


def test_一様ランダムなデータ収集方策():
    # Arrange
    context = np.array([[1.0, 0.0], [0.0, 1.0]])
    action_context = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])

    # Act
    action_dist = random_policy(context, action_context)

    # Assert
    assert action_dist.shape == (
        2,
        3,
    ), "出力値ののshapeは(ラウンド数, アクション数)である"
    assert np.all(
        action_dist >= 0.0
    ), "任意のラウンドで、各アクションの選択確率は0以上である"
    assert np.all(
        action_dist == 1.0 / 3
    ), "任意のラウンドで、各アクションの選択確率は、一様に1/アクション数である"
