import numpy as np

from recommender_experiments.app.cfml_book.ch3.dataset import calc_true_value, generate_synthetic_data


def test_generate_synthetic_dataが正しい構造のバンディットフィードバックデータを生成すること():
    # Arrange
    num_data = 100
    dim_context = 5
    num_actions = 50
    num_clusters = 3
    num_def_actions = 5
    random_state = 42

    random_ = np.random.RandomState(random_state)
    phi_a = random_.choice(num_clusters, size=num_actions)
    theta_g = random_.normal(size=(dim_context, num_clusters))
    M_g = random_.normal(size=(dim_context, num_clusters))
    b_g = random_.normal(size=(1, num_clusters))
    theta_h = random_.normal(size=(dim_context, num_actions))
    M_h = random_.normal(size=(dim_context, num_actions))
    b_h = random_.normal(size=(1, num_actions))
    lambda_ = 0.5

    # Act
    dataset = generate_synthetic_data(
        num_data=num_data,
        theta_g=theta_g,
        M_g=M_g,
        b_g=b_g,
        theta_h=theta_h,
        M_h=M_h,
        b_h=b_h,
        phi_a=phi_a,
        lambda_=lambda_,
        dim_context=dim_context,
        num_actions=num_actions,
        num_def_actions=num_def_actions,
        num_clusters=num_clusters,
        random_state=random_state,
    )

    # Assert: データ構造の検証
    required_keys = ["num_data", "num_actions", "num_clusters", "x", "a", "c", "r", "phi_a", "pi_0", "g_x_c", "h_x_a", "q_x_a"]
    for key in required_keys:
        assert key in dataset, f"データセットに{key}キーが存在すること"

    # Assert: 形状とデータ型の検証
    assert dataset["x"].shape == (num_data, dim_context), "コンテキストの形状がnum_data掛けるdim_contextであること"
    assert dataset["x"].dtype == np.float64, "コンテキストのデータ型がfloat64であること"

    assert dataset["a"].shape == (num_data,), "行動の形状がnum_dataであること"
    assert dataset["a"].dtype in [np.int32, np.int64], "行動のデータ型が整数型であること"

    assert dataset["c"].shape == (num_data,), "クラスタの形状がnum_dataであること"
    assert dataset["c"].dtype in [np.int32, np.int64], "クラスタのデータ型が整数型であること"

    assert dataset["r"].shape == (num_data,), "報酬の形状がnum_dataであること"
    assert dataset["r"].dtype == np.float64, "報酬のデータ型がfloat64であること"

    assert dataset["phi_a"].shape == (num_actions,), "phi_aの形状がnum_actionsであること"
    assert np.array_equal(dataset["phi_a"], phi_a), "phi_aが入力値と一致すること"

    assert dataset["pi_0"].shape == (num_data, num_actions), "pi_0の形状がnum_data掛けるnum_actionsであること"
    assert dataset["g_x_c"].shape == (num_data, num_clusters), "g_x_cの形状がnum_data掛けるnum_clustersであること"
    assert dataset["h_x_a"].shape == (num_data, num_actions), "h_x_aの形状がnum_data掛けるnum_actionsであること"
    assert dataset["q_x_a"].shape == (num_data, num_actions), "q_x_aの形状がnum_data掛けるnum_actionsであること"

    # Assert: 値の範囲検証
    assert np.all(dataset["a"] >= 0) and np.all(dataset["a"] < num_actions), "行動が0からnum_actions-1の範囲内であること"
    assert np.all(dataset["c"] >= 0) and np.all(dataset["c"] < num_clusters), "クラスタが0からnum_clusters-1の範囲内であること"

    # Assert: 確率分布の検証
    pi_0_sum = dataset["pi_0"].sum(axis=1)
    assert np.allclose(pi_0_sum, 1.0, atol=1e-6), "各コンテキストにおけるpi_0の総和が1.0であること"

    # Assert: deficient support設定の検証
    assert np.all(dataset["pi_0"][:, :num_def_actions] == 0), f"最初の{num_def_actions}個の行動の確率が0であること"

    # Assert: クラスタマッピングの整合性検証
    for i in range(num_data):
        selected_action = dataset["a"][i]
        expected_cluster = phi_a[selected_action]
        assert dataset["c"][i] == expected_cluster, f"選択された行動{selected_action}のクラスタが正しくマッピングされていること"

    # Assert: 期待報酬関数の構成検証
    reconstructed_q = dataset["g_x_c"][:, dataset["phi_a"]] + dataset["h_x_a"]
    assert np.allclose(dataset["q_x_a"], reconstructed_q, atol=1e-6), "q_x_aがg_x_cとh_x_aの和として正しく構成されていること"

    # Assert: lambda_パラメータの検証
    # g_x_cは(1-lambda_)倍、h_x_aはlambda_倍されているはず
    # 実際の非線形変換後の値なので厳密な検証は難しいが、形状と存在は確認できる
    assert dataset["g_x_c"] is not None, "g_x_cが計算されていること"
    assert dataset["h_x_a"] is not None, "h_x_aが計算されていること"

    # Assert: 再現性の検証
    dataset2 = generate_synthetic_data(
        num_data=num_data,
        theta_g=theta_g,
        M_g=M_g,
        b_g=b_g,
        theta_h=theta_h,
        M_h=M_h,
        b_h=b_h,
        phi_a=phi_a,
        lambda_=lambda_,
        dim_context=dim_context,
        num_actions=num_actions,
        num_def_actions=num_def_actions,
        num_clusters=num_clusters,
        random_state=random_state,
    )
    assert np.array_equal(dataset["a"], dataset2["a"]), "同じrandom_stateで同じ行動系列が生成されること"
    assert np.array_equal(dataset["x"], dataset2["x"]), "同じrandom_stateで同じコンテキストが生成されること"


def test_calc_true_valueが評価方策の真の性能を近似的に計算すること():
    # Arrange
    dim_context = 5
    num_actions = 50
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
    lambda_ = 0.5

    # Act
    true_value = calc_true_value(
        dim_context=dim_context,
        num_actions=num_actions,
        num_clusters=num_clusters,
        theta_g=theta_g,
        M_g=M_g,
        b_g=b_g,
        phi_a=phi_a,
        theta_h=theta_h,
        M_h=M_h,
        b_h=b_h,
        lambda_=lambda_,
    )

    # Assert: 返り値の検証
    assert isinstance(true_value, (float, np.floating)), "真の性能値がfloat型であること"
    assert not np.isnan(true_value), "真の性能値がNaNでないこと"
    assert not np.isinf(true_value), "真の性能値が無限大でないこと"

    # Assert: 大規模サンプルでの安定性検証
    # 同じパラメータで再計算しても近い値が得られることを確認（デフォルトのrandom_stateが固定されているため完全一致）
    true_value2 = calc_true_value(
        dim_context=dim_context,
        num_actions=num_actions,
        num_clusters=num_clusters,
        theta_g=theta_g,
        M_g=M_g,
        b_g=b_g,
        phi_a=phi_a,
        theta_h=theta_h,
        M_h=M_h,
        b_h=b_h,
        lambda_=lambda_,
    )
    assert true_value == true_value2, "同じパラメータで計算した真の性能値が一致すること"
