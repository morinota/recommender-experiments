import numpy as np

from recommender_experiments.service.environment.ranking_synthetic_dataset import RankingSyntheticBanditDataset


def test_ランキング問題用の合成バンディットフィードバックデータを生成できること():
    # Arrange
    num_data = 3
    dim_context = 4
    dim_action_context = 6
    num_actions = 5
    k = 3
    theta = np.random.normal(size=(dim_context, num_actions))
    M = np.random.normal(size=(dim_context, num_actions))
    b = np.random.normal(size=(num_actions, 1))
    W = np.random.normal(size=(k, k))
    beta = -1.0
    reward_noise = 0.5
    p = [0.8, 0.1, 0.1]
    p_rand = 0.2
    action_context = np.random.randn(num_actions, dim_action_context)

    sut = RankingSyntheticBanditDataset(
        dim_context=dim_context,
        num_actions=num_actions,
        k=k,
        theta=theta,
        M=M,
        b=b,
        W=W,
        beta=beta,
        reward_noise=reward_noise,
        p=p,
        p_rand=p_rand,
        action_context=action_context,
        random_state=12345,
        is_test=True,  # テストモード
    )

    # Act
    bandit_feedaback = sut.obtain_batch_bandit_feedback(num_data)

    # Assert
    assert True
