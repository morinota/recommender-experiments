import numpy as np

from recommender_experiments.app.cfml_book.ch5.dataset import generate_synthetic_data


def test_generate_synthetic_dataが正しい構造のバンディットフィードバックデータを生成すること():
    # Arrange
    num_data = 100
    dim_context = 3
    num_actions = 10
    num_clusters = 3
    random_state = 42

    random_ = np.random.RandomState(random_state)
    phi_a = random_.choice(num_clusters, size=num_actions)
    theta_g = random_.normal(size=(dim_context, num_clusters))
    M_g = random_.normal(size=(dim_context, num_clusters))
    b_g = random_.normal(size=(1, num_clusters))
    theta_h = random_.normal(size=(dim_context, num_actions))
    M_h = random_.normal(size=(dim_context, num_actions))
    b_h = random_.normal(size=(1, num_actions))

    # Act
    dataset = generate_synthetic_data(
        num_data=num_data,
        dim_context=dim_context,
        num_actions=num_actions,
        num_clusters=num_clusters,
        phi_a=phi_a,
        theta_g=theta_g,
        M_g=M_g,
        b_g=b_g,
        theta_h=theta_h,
        M_h=M_h,
        b_h=b_h,
        random_state=random_state,
    )

    # Assert: データ構造の検証
    required_keys = [
        "num_data",
        "num_actions",
        "num_clusters",
        "x",
        "a",
        "c",
        "r",
        "phi_a",
        "pi_0",
        "pi_0_c",
        "pscore",
        "pscore_c",
        "g_x_c",
        "h_x_a",
        "q_x_a",
    ]
    for key in required_keys:
        assert key in dataset, f"{key}がログデータに含まれること"

    # Assert: 形状とデータ型の検証
    assert dataset["x"].shape == (num_data, dim_context), "コンテキストxの形状が正しいこと"
    assert len(dataset["a"]) == num_data, "行動aの数がnum_dataと一致すること"
    assert len(dataset["r"]) == num_data, "報酬rの数がnum_dataと一致すること"

    # Assert: 値の範囲検証
    assert np.all(dataset["a"] >= 0) and np.all(dataset["a"] < num_actions), "行動aが有効な範囲内であること"
    assert np.all(np.isin(dataset["r"], [0, 1])), "報酬rが0または1のバイナリ値であること"
    assert np.all(dataset["q_x_a"] >= 0) and np.all(dataset["q_x_a"] <= 1), "期待報酬q_x_aが0-1の範囲内であること"

    # Assert: 確率分布の検証
    pi_0_sum = dataset["pi_0"].sum(axis=1)
    assert np.allclose(pi_0_sum, 1.0, atol=1e-6), "各コンテキストにおけるpi_0の総和が1.0であること"

    pi_0_c_sum = dataset["pi_0_c"].sum(axis=1)
    assert np.allclose(pi_0_c_sum, 1.0, atol=1e-6), "各コンテキストにおけるpi_0_cの総和が1.0であること"

    # Assert: pscoreの整合性検証
    idx = np.arange(num_data)
    expected_pscore = dataset["pi_0"][idx, dataset["a"]]
    assert np.allclose(dataset["pscore"], expected_pscore), "pscoreが選択された行動aのpi_0と一致すること"

    # Assert: 期待報酬の構成検証
    g_contribution = dataset["g_x_c"][:, phi_a]
    h_contribution = dataset["h_x_a"]
    reconstructed_q = g_contribution + h_contribution
    assert np.allclose(dataset["q_x_a"], reconstructed_q, atol=1e-10), "q_x_aがg_x_cとh_x_aの和であること"

    # Assert: クラスタマッピングの検証
    assert len(dataset["phi_a"]) == num_actions, "phi_aの長さがnum_actionsと一致すること"
    assert np.all(dataset["phi_a"] >= 0) and np.all(
        dataset["phi_a"] < num_clusters
    ), "phi_aが有効なクラスタID範囲内であること"

    expected_c = dataset["phi_a"][dataset["a"]]
    assert np.array_equal(dataset["c"], expected_c), "cがphi_a[a]と一致すること"

    # Assert: 再現性の検証
    dataset2 = generate_synthetic_data(
        num_data=num_data,
        dim_context=dim_context,
        num_actions=num_actions,
        num_clusters=num_clusters,
        phi_a=phi_a,
        theta_g=theta_g,
        M_g=M_g,
        b_g=b_g,
        theta_h=theta_h,
        M_h=M_h,
        b_h=b_h,
        random_state=random_state,
    )
    assert np.array_equal(dataset["x"], dataset2["x"]), "同じrandom_stateでコンテキストxが再現されること"
    assert np.array_equal(dataset["a"], dataset2["a"]), "同じrandom_stateで行動aが再現されること"
    assert np.array_equal(dataset["r"], dataset2["r"]), "同じrandom_stateで報酬rが再現されること"

    # Assert: lambda値による構成比の検証
    dataset_cluster_only = generate_synthetic_data(
        num_data=num_data,
        dim_context=dim_context,
        num_actions=num_actions,
        num_clusters=num_clusters,
        phi_a=phi_a,
        theta_g=theta_g,
        M_g=M_g,
        b_g=b_g,
        theta_h=theta_h,
        M_h=M_h,
        b_h=b_h,
        lambda_=0.0,
        random_state=random_state,
    )
    assert np.allclose(dataset_cluster_only["h_x_a"], 0.0, atol=1e-10), "lambda=0の時h_x_aが0であること"

    dataset_residual_only = generate_synthetic_data(
        num_data=num_data,
        dim_context=dim_context,
        num_actions=num_actions,
        num_clusters=num_clusters,
        phi_a=phi_a,
        theta_g=theta_g,
        M_g=M_g,
        b_g=b_g,
        theta_h=theta_h,
        M_h=M_h,
        b_h=b_h,
        lambda_=1.0,
        random_state=random_state,
    )
    assert np.allclose(dataset_residual_only["g_x_c"], 0.0, atol=1e-10), "lambda=1の時g_x_cが0であること"
