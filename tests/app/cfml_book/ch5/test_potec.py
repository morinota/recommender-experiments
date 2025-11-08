import numpy as np

from recommender_experiments.app.cfml_book.ch5.potec import POTEC


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
