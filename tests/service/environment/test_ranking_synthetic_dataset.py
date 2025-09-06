import numpy as np

from recommender_experiments.service.environment.ranking_synthetic_dataset import (
    RankingSyntheticBanditDataset,
    SyntheticRankingData,
)


def test_合成データ生成の基本動作():
    """RankingSyntheticBanditDatasetが正しくSyntheticRankingDataを返し、基本的な制約を満たすことをテストする。"""
    # Arrange
    num_data = 10
    dim_context = 3
    num_actions = 5
    K = 2
    theta = np.random.normal(size=(dim_context, num_actions))
    M = np.random.normal(size=(dim_context, num_actions))
    b = np.random.normal(size=(num_actions, 1))
    W = np.random.normal(size=(K, K))
    action_context = np.random.normal(size=(num_actions, 6))
    random_state = 12345

    dataset = RankingSyntheticBanditDataset(
        dim_context=dim_context,
        num_actions=num_actions,
        k=K,
        theta=theta,
        M=M,
        b=b,
        W=W,
        action_context=action_context,
        random_state=random_state,
    )

    # Act
    result = dataset.obtain_batch_bandit_feedback(num_data)

    # Assert - 返り値の型と基本属性
    assert isinstance(result, SyntheticRankingData)
    assert result.num_data == num_data
    assert result.K == K
    assert result.num_actions == num_actions

    # Assert - データの形状
    assert result.x.shape == (num_data, dim_context)
    assert result.a_k.shape == (num_data, K)
    assert result.r_k.shape == (num_data, K)
    assert result.C.shape == (num_data, K, K)
    assert result.pi_0.shape == (num_data, num_actions)
    assert result.q_k.shape == (num_data, K)
    assert result.base_q_func.shape == (num_data, num_actions)

    # Assert - データの値域制約
    assert np.all(result.C >= 0) and np.all(result.C <= 1)  # ユーザ行動行列は[0,1]
    assert np.all(result.a_k >= 0) and np.all(result.a_k < num_actions)  # 行動は有効範囲内
    assert np.allclose(result.pi_0.sum(axis=1), 1.0, atol=1e-6)  # 方策は確率分布
    assert np.all(result.base_q_func >= 0) and np.all(result.base_q_func <= 1)  # sigmoid出力


def test_再現性が保たれること():
    """同じrandom_stateを使用した場合に同じ結果が得られることをテストする。"""
    # Arrange
    num_data = 5
    dim_context = 2
    num_actions = 3
    K = 2
    theta = np.random.normal(size=(dim_context, num_actions))
    M = np.random.normal(size=(dim_context, num_actions))
    b = np.random.normal(size=(num_actions, 1))
    W = np.random.normal(size=(K, K))
    action_context = np.random.normal(size=(num_actions, 4))
    random_state = 999

    dataset_params = {
        "dim_context": dim_context,
        "num_actions": num_actions,
        "k": K,
        "theta": theta,
        "M": M,
        "b": b,
        "W": W,
        "action_context": action_context,
        "random_state": random_state,
    }

    dataset1 = RankingSyntheticBanditDataset(**dataset_params)
    dataset2 = RankingSyntheticBanditDataset(**dataset_params)

    # Act
    result1 = dataset1.obtain_batch_bandit_feedback(num_data)
    result2 = dataset2.obtain_batch_bandit_feedback(num_data)

    # Assert - すべてのデータが一致
    assert np.array_equal(result1.x, result2.x)
    assert np.array_equal(result1.a_k, result2.a_k)
    assert np.array_equal(result1.r_k, result2.r_k)
    assert np.array_equal(result1.C, result2.C)
    assert np.array_equal(result1.pi_0, result2.pi_0)
    assert np.array_equal(result1.q_k, result2.q_k)
    assert np.array_equal(result1.base_q_func, result2.base_q_func)
