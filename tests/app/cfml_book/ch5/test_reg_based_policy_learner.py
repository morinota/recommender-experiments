import numpy as np

from recommender_experiments.app.cfml_book.ch5.reg_based_policy_learner import RegBasedPolicyLearner


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
