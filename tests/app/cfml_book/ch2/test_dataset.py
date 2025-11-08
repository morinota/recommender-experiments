import numpy as np

from recommender_experiments.app.cfml_book.ch2.dataset import calc_true_value, generate_synthetic_data


def test_generate_synthetic_dataが正しい構造のランキングバンディットフィードバックデータを生成すること():
    # Arrange
    num_data = 100
    dim_context = 3
    num_actions = 10
    K = 3  # ランキングサイズ
    random_state = 42

    random_ = np.random.RandomState(random_state)
    theta = random_.normal(size=(dim_context, num_actions))
    M = random_.normal(size=(dim_context, num_actions))
    b = random_.normal(size=(1, num_actions))
    W = random_.uniform(0.5, 1.5, size=(K, K))

    # Act
    dataset = generate_synthetic_data(
        num_data=num_data,
        dim_context=dim_context,
        num_actions=num_actions,
        K=K,
        theta=theta,
        M=M,
        b=b,
        W=W,
        random_state=random_state,
    )

    # Assert: データ構造の検証
    required_keys = ["num_data", "K", "num_actions", "x", "a_k", "r_k", "C", "pi_0", "q_k", "base_q_func"]
    for key in required_keys:
        assert key in dataset, f"{key}がログデータに含まれること"

    # Assert: 形状とデータ型の検証
    assert dataset["x"].shape == (num_data, dim_context), "コンテキストxの形状が正しいこと"
    assert dataset["a_k"].shape == (num_data, K), "ランキングa_kの形状が正しいこと"
    assert dataset["r_k"].shape == (num_data, K), "報酬r_kの形状が正しいこと"
    assert dataset["C"].shape == (num_data, K, K), "ユーザ行動行列Cの形状が正しいこと"
    assert dataset["pi_0"].shape == (num_data, num_actions), "方策pi_0の形状が正しいこと"
    assert dataset["q_k"].shape == (num_data, K), "期待報酬q_kの形状が正しいこと"

    # Assert: 値の範囲検証
    assert np.all(dataset["a_k"] >= 0) and np.all(dataset["a_k"] < num_actions), "ランキング内の行動が有効な範囲内であること"
    assert np.all(dataset["C"] >= 0) and np.all(dataset["C"] <= 1), "ユーザ行動行列Cが0-1の範囲内であること"
    assert np.all(dataset["base_q_func"] >= 0) and np.all(
        dataset["base_q_func"] <= 1
    ), "基本期待報酬が0-1の範囲内であること"

    # Assert: 確率分布の検証
    pi_0_sum = dataset["pi_0"].sum(axis=1)
    assert np.allclose(pi_0_sum, 1.0, atol=1e-6), "各コンテキストにおけるpi_0の総和が1.0であること"

    # Assert: 再現性の検証
    dataset2 = generate_synthetic_data(
        num_data=num_data,
        dim_context=dim_context,
        num_actions=num_actions,
        K=K,
        theta=theta,
        M=M,
        b=b,
        W=W,
        random_state=random_state,
    )
    assert np.array_equal(dataset["x"], dataset2["x"]), "同じrandom_stateでコンテキストxが再現されること"
    assert np.array_equal(dataset["a_k"], dataset2["a_k"]), "同じrandom_stateでランキングa_kが再現されること"

    # Assert: ユーザ行動モデルpのバリエーション検証
    dataset_independent = generate_synthetic_data(
        num_data=num_data,
        dim_context=dim_context,
        num_actions=num_actions,
        K=K,
        theta=theta,
        M=M,
        b=b,
        W=W,
        p=[1.0, 0.0, 0.0],  # independent only
        random_state=random_state,
    )
    # independentの場合、Cは単位行列（対角成分のみ1）
    for i in range(num_data):
        assert np.allclose(np.diag(dataset_independent["C"][i]), 1.0), "independentモデルで対角成分が1であること"

    dataset_cascade = generate_synthetic_data(
        num_data=num_data,
        dim_context=dim_context,
        num_actions=num_actions,
        K=K,
        theta=theta,
        M=M,
        b=b,
        W=W,
        p=[0.0, 1.0, 0.0],  # cascade only
        random_state=random_state,
    )
    # cascadeの場合、Cは下三角行列
    for i in range(num_data):
        assert np.allclose(dataset_cascade["C"][i], np.tril(dataset_cascade["C"][i])), "cascadeモデルで下三角であること"

    # Assert: is_testフラグによる方策の切り替え検証
    dataset_test = generate_synthetic_data(
        num_data=num_data,
        dim_context=dim_context,
        num_actions=num_actions,
        K=K,
        theta=theta,
        M=M,
        b=b,
        W=W,
        is_test=True,
        random_state=random_state,
    )
    # is_test=Trueの場合はeps_greedy_policyが使われる
    # 確率分布の形状が異なるはず（softmaxではない）
    assert not np.allclose(dataset["pi_0"], dataset_test["pi_0"]), "is_testフラグで方策が変わること"


def test_calc_true_valueが評価方策の真の性能を近似的に計算すること():
    # Arrange
    dim_context = 3
    num_actions = 10
    K = 3
    random_state = 42

    random_ = np.random.RandomState(random_state)
    theta = random_.normal(size=(dim_context, num_actions))
    M = random_.normal(size=(dim_context, num_actions))
    b = random_.normal(size=(1, num_actions))
    W = random_.uniform(0.5, 1.5, size=(K, K))

    # Act
    true_value = calc_true_value(
        dim_context=dim_context,
        num_actions=num_actions,
        K=K,
        p=[0.5, 0.3, 0.2],
        theta=theta,
        M=M,
        b=b,
        W=W,
        num_data=1000,
    )

    # Assert: 真の性能が数値として得られること
    assert isinstance(true_value, (float, np.floating)), "真の性能がfloat型であること"
    assert not np.isnan(true_value), "真の性能がNaNでないこと"
    assert not np.isinf(true_value), "真の性能が無限大でないこと"

    # Assert: 大きなサンプル数で計算すると安定した値になること
    true_value_large = calc_true_value(
        dim_context=dim_context,
        num_actions=num_actions,
        K=K,
        p=[0.5, 0.3, 0.2],
        theta=theta,
        M=M,
        b=b,
        W=W,
        num_data=10000,
    )
    # より大きなサンプル数で計算した値と近いはず
    assert abs(true_value_large - true_value) < abs(
        true_value
    ), "大きなサンプル数で計算した値が小さなサンプル数の値に近いこと"
