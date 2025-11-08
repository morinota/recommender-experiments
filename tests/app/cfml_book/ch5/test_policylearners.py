import numpy as np
import pytest
import torch

from recommender_experiments.app.cfml_book.ch5.dataset import generate_synthetic_data
from recommender_experiments.app.cfml_book.ch5.policylearners import (
    POTEC,
    GradientBasedPolicyLearner,
    RegBasedPolicyLearner,
)


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


# RegBasedPolicyLearnerのテスト


def test_RegBasedPolicyLearner初期化時に正しいプロパティが設定されること():
    # Arrange
    dim_x = 5
    num_actions = 10

    # Act
    sut = RegBasedPolicyLearner(dim_x=dim_x, num_actions=num_actions, max_iter=5)

    # Assert
    assert sut.dim_x == dim_x, "dim_xが正しく設定されること"
    assert sut.num_actions == num_actions, "num_actionsが正しく設定されること"
    assert sut.nn_model is not None, "NNモデルが初期化されること"
    assert len(sut.train_loss) == 0, "train_lossが空のリストで初期化されること"
    assert len(sut.train_value) == 0, "train_valueが空のリストで初期化されること"
    assert len(sut.test_value) == 0, "test_valueが空のリストで初期化されること"


def test_RegBasedPolicyLearner異なるactivation関数で初期化できること():
    # Arrange & Act & Assert
    for activation in ["tanh", "relu", "elu"]:
        sut = RegBasedPolicyLearner(dim_x=5, num_actions=10, activation=activation)
        assert sut.nn_model is not None, f"activation={activation}でモデルが初期化されること"


def test_RegBasedPolicyLearner異なるsolverで初期化できること():
    # Arrange & Act & Assert
    for solver in ["adagrad", "adam"]:
        sut = RegBasedPolicyLearner(dim_x=5, num_actions=10, solver=solver, max_iter=1)
        assert sut.solver == solver, f"solver={solver}が設定されること"


def test_RegBasedPolicyLearnerのfit後に学習履歴が記録されること(small_synthetic_dataset):
    # Arrange
    dataset, dim_x, num_actions, _ = small_synthetic_dataset
    max_iter = 3
    sut = RegBasedPolicyLearner(dim_x=dim_x, num_actions=num_actions, max_iter=max_iter, batch_size=8)

    # Act
    sut.fit(dataset, dataset)

    # Assert
    assert len(sut.train_loss) == max_iter, f"train_lossが{max_iter}エポック分記録されること"
    assert len(sut.train_value) == max_iter, f"train_valueが{max_iter}エポック分記録されること"
    assert len(sut.test_value) == max_iter, f"test_valueが{max_iter}エポック分記録されること"
    assert all(isinstance(v, float) for v in sut.train_loss), "train_lossが全てfloat型であること"


def test_RegBasedPolicyLearnerのpredict時に正しい形状の方策が返されること(small_synthetic_dataset):
    # Arrange
    dataset, dim_x, num_actions, _ = small_synthetic_dataset
    sut = RegBasedPolicyLearner(dim_x=dim_x, num_actions=num_actions, max_iter=2)
    sut.fit(dataset, dataset)

    # Act
    pi = sut.predict(dataset)

    # Assert
    assert pi.shape == (dataset["num_data"], num_actions), "方策の形状が(n_samples, n_actions)であること"
    assert np.allclose(pi.sum(axis=1), 1.0), "各サンプルの方策の総和が1.0であること"
    assert np.all(pi >= 0), "全ての確率が非負であること"


def test_RegBasedPolicyLearnerのpredict_q時に正しい形状のQ値が返されること(small_synthetic_dataset):
    # Arrange
    dataset, dim_x, num_actions, _ = small_synthetic_dataset
    sut = RegBasedPolicyLearner(dim_x=dim_x, num_actions=num_actions, max_iter=2)
    sut.fit(dataset, dataset)

    # Act
    q_hat = sut.predict_q(dataset)

    # Assert
    assert q_hat.shape == (dataset["num_data"], num_actions), "Q値の形状が(n_samples, n_actions)であること"


# GradientBasedPolicyLearnerのテスト


def test_GradientBasedPolicyLearner初期化時に正しいプロパティが設定されること():
    # Arrange
    dim_x = 5
    num_actions = 10

    # Act
    sut = GradientBasedPolicyLearner(dim_x=dim_x, num_actions=num_actions, max_iter=5)

    # Assert
    assert sut.dim_x == dim_x, "dim_xが正しく設定されること"
    assert sut.num_actions == num_actions, "num_actionsが正しく設定されること"
    assert sut.nn_model is not None, "NNモデルが初期化されること"
    assert len(sut.train_loss) == 0, "train_lossが空のリストで初期化されること"


def test_GradientBasedPolicyLearnerのfit後に学習履歴が記録されること(small_synthetic_dataset):
    # Arrange
    dataset, dim_x, num_actions, _ = small_synthetic_dataset
    max_iter = 3
    sut = GradientBasedPolicyLearner(dim_x=dim_x, num_actions=num_actions, max_iter=max_iter, batch_size=8)

    # Act
    sut.fit(dataset, dataset)

    # Assert
    assert len(sut.train_loss) == max_iter, f"train_lossが{max_iter}エポック分記録されること"
    assert len(sut.train_value) == max_iter, f"train_valueが{max_iter}エポック分記録されること"
    assert len(sut.test_value) == max_iter, f"test_valueが{max_iter}エポック分記録されること"


def test_GradientBasedPolicyLearnerのfit時にq_hatを指定できること(small_synthetic_dataset):
    # Arrange
    dataset, dim_x, num_actions, _ = small_synthetic_dataset
    sut = GradientBasedPolicyLearner(dim_x=dim_x, num_actions=num_actions, max_iter=2, batch_size=8)
    q_hat = dataset["q_x_a"] + np.random.normal(scale=0.1, size=dataset["q_x_a"].shape)

    # Act
    sut.fit(dataset, dataset, q_hat=q_hat)

    # Assert
    assert len(sut.test_value) == 2, "q_hatを指定してもfitが正常に完了すること"


def test_GradientBasedPolicyLearnerのpredict時に正しい形状の方策が返されること(small_synthetic_dataset):
    # Arrange
    dataset, dim_x, num_actions, _ = small_synthetic_dataset
    sut = GradientBasedPolicyLearner(dim_x=dim_x, num_actions=num_actions, max_iter=2)
    sut.fit(dataset, dataset)

    # Act
    pi = sut.predict(dataset)

    # Assert
    assert pi.shape == (dataset["num_data"], num_actions), "方策の形状が(n_samples, n_actions)であること"
    assert np.allclose(pi.sum(axis=1), 1.0, atol=1e-5), "各サンプルの方策の総和が1.0であること"
    assert np.all(pi >= 0), "全ての確率が非負であること"


# POTECのテスト


def test_POTEC初期化時に正しいプロパティが設定されること():
    # Arrange
    dim_x = 5
    num_actions = 10
    num_clusters = 3

    # Act
    sut = POTEC(dim_x=dim_x, num_actions=num_actions, num_clusters=num_clusters, max_iter=5)

    # Assert
    assert sut.dim_x == dim_x, "dim_xが正しく設定されること"
    assert sut.num_actions == num_actions, "num_actionsが正しく設定されること"
    assert sut.num_clusters == num_clusters, "num_clustersが正しく設定されること"
    assert sut.nn_model is not None, "NNモデルが初期化されること"


def test_POTECのfit後に学習履歴が記録されること(small_synthetic_dataset):
    # Arrange
    dataset, dim_x, num_actions, num_clusters = small_synthetic_dataset
    max_iter = 3
    sut = POTEC(dim_x=dim_x, num_actions=num_actions, num_clusters=num_clusters, max_iter=max_iter, batch_size=8)

    f_hat = dataset["h_x_a"] + np.random.normal(scale=0.1, size=dataset["h_x_a"].shape)

    # Act
    sut.fit(dataset, dataset, f_hat=f_hat, f_hat_test=f_hat)

    # Assert
    assert len(sut.train_loss) == max_iter, f"train_lossが{max_iter}エポック分記録されること"
    assert len(sut.train_value) == max_iter, f"train_valueが{max_iter}エポック分記録されること"
    assert len(sut.test_value) == max_iter, f"test_valueが{max_iter}エポック分記録されること"


def test_POTECのpredict時に正しい形状の方策が返されること(small_synthetic_dataset):
    # Arrange
    dataset, dim_x, num_actions, num_clusters = small_synthetic_dataset
    sut = POTEC(dim_x=dim_x, num_actions=num_actions, num_clusters=num_clusters, max_iter=2)
    f_hat = dataset["h_x_a"] + np.random.normal(scale=0.1, size=dataset["h_x_a"].shape)
    sut.fit(dataset, dataset, f_hat=f_hat, f_hat_test=f_hat)

    # Act
    pi = sut.predict(dataset, f_hat)

    # Assert
    assert pi.shape == (dataset["num_data"], num_actions), "方策の形状が(n_samples, n_actions)であること"
    assert np.allclose(pi.sum(axis=1), 1.0, atol=1e-5), "各サンプルの方策の総和が1.0であること"
    assert np.all(pi >= 0), "全ての確率が非負であること"


def test_POTECのpredict時にクラスタベース方策が正しく生成されること(small_synthetic_dataset):
    # Arrange
    dataset, dim_x, num_actions, num_clusters = small_synthetic_dataset
    sut = POTEC(dim_x=dim_x, num_actions=num_actions, num_clusters=num_clusters, max_iter=2)
    f_hat = dataset["h_x_a"] + np.random.normal(scale=0.1, size=dataset["h_x_a"].shape)
    sut.fit(dataset, dataset, f_hat=f_hat, f_hat_test=f_hat)

    # Act
    pi = sut.predict(dataset, f_hat)

    # Assert
    # 各サンプルで非ゼロの確率を持つアクション数はクラスタ数以下
    non_zero_actions_per_sample = (pi > 0).sum(axis=1)
    assert np.all(non_zero_actions_per_sample <= num_clusters), (
        "各サンプルで選択されるアクションはクラスタ数以下であること"
    )
