import numpy as np

from recommender_experiments.app.cfml_book.ch5.gradient_based_policy_learner import GradientBasedPolicyLearner


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
