import numpy as np
import pytest

from recommender_experiments.app.cfml_book.ch5.dataset import generate_synthetic_data


@pytest.fixture
def small_synthetic_dataset():
    """小規模な合成データセット（テスト用に軽量化）"""
    num_data = 50
    dim_x = 3
    num_actions = 10
    num_clusters = 3
    lambda_ = 0.5
    random_state = 12345

    np.random.seed(random_state)
    phi_a = np.random.choice(num_clusters, size=num_actions)
    theta_g = np.random.normal(size=(dim_x, num_clusters))
    M_g = np.random.normal(size=(dim_x, num_clusters))
    b_g = np.random.normal(size=(1, num_clusters))
    theta_h = np.random.normal(size=(dim_x, num_actions))
    M_h = np.random.normal(size=(dim_x, num_actions))
    b_h = np.random.normal(size=(1, num_actions))

    dataset = generate_synthetic_data(
        num_data=num_data,
        lambda_=lambda_,
        theta_g=theta_g,
        M_g=M_g,
        b_g=b_g,
        theta_h=theta_h,
        M_h=M_h,
        b_h=b_h,
        phi_a=phi_a,
        dim_context=dim_x,
        num_actions=num_actions,
        num_clusters=num_clusters,
        random_state=random_state,
    )

    return dataset, dim_x, num_actions, num_clusters
