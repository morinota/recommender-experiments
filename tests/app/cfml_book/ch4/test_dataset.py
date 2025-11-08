import numpy as np

from recommender_experiments.app.cfml_book.ch4.dataset import calc_true_value, generate_synthetic_data


def test_generate_synthetic_dataが正しい構造のエピソディックデータを生成すること():
    # Arrange
    num_data = 50
    dim_state = 3
    num_states = 10
    num_actions = 5
    H = 4
    random_state = 42

    random_ = np.random.RandomState(random_state)
    theta = random_.normal(size=(dim_state, num_actions))
    M = random_.normal(size=(dim_state, num_actions))
    b = random_.normal(size=(1, num_actions))
    init_dist = random_.dirichlet(np.ones(num_states))
    trans_probs = random_.dirichlet(np.ones(num_states), size=(num_states, num_actions))
    trans_probs = trans_probs.transpose(0, 2, 1)  # (S, A, S) -> (S, S, A)

    # Act
    dataset = generate_synthetic_data(
        num_data=num_data,
        dim_state=dim_state,
        num_states=num_states,
        num_actions=num_actions,
        H=H,
        theta=theta,
        M=M,
        b=b,
        init_dist=init_dist,
        trans_probs=trans_probs,
        random_state=random_state,
    )

    # Assert: データ構造の検証
    required_keys = ["num_data", "H", "num_states", "num_actions", "s_h", "a_h", "r_h", "S", "pi_0", "pi", "q_h", "q_s_a"]
    for key in required_keys:
        assert key in dataset, f"データセットに{key}キーが存在すること"

    # Assert: 形状とデータ型の検証
    assert dataset["s_h"].shape == (num_data, H), "状態軌跡の形状がnum_data掛けるHであること"
    assert dataset["s_h"].dtype in [np.int32, np.int64], "状態軌跡のデータ型が整数型であること"

    assert dataset["a_h"].shape == (num_data, H), "行動軌跡の形状がnum_data掛けるHであること"
    assert dataset["a_h"].dtype in [np.int32, np.int64], "行動軌跡のデータ型が整数型であること"

    assert dataset["r_h"].shape == (num_data, H), "報酬軌跡の形状がnum_data掛けるHであること"
    assert dataset["r_h"].dtype == np.float64, "報酬軌跡のデータ型がfloat64であること"

    assert dataset["S"].shape == (num_states, dim_state), "状態埋め込み行列の形状がnum_states掛けるdim_stateであること"
    assert dataset["S"].dtype == np.float64, "状態埋め込み行列のデータ型がfloat64であること"

    assert dataset["pi_0"].shape == (num_data, num_actions, H), "データ収集方策の形状がnum_data掛けるnum_actions掛けるHであること"
    assert dataset["pi"].shape == (num_data, num_actions, H), "評価方策の形状がnum_data掛けるnum_actions掛けるHであること"

    assert dataset["q_h"].shape == (num_data, H), "期待報酬軌跡の形状がnum_data掛けるHであること"
    assert dataset["q_s_a"].shape == (num_states, num_actions), "状態価値関数の形状がnum_states掛けるnum_actionsであること"

    # Assert: メタデータの検証
    assert dataset["num_data"] == num_data, "num_dataが正しく設定されていること"
    assert dataset["H"] == H, "Hが正しく設定されていること"
    assert dataset["num_states"] == num_states, "num_statesが正しく設定されていること"
    assert dataset["num_actions"] == num_actions, "num_actionsが正しく設定されていること"

    # Assert: 値の範囲検証
    assert np.all(dataset["s_h"] >= 0) and np.all(dataset["s_h"] < num_states), "状態が0からnum_states-1の範囲内であること"
    assert np.all(dataset["a_h"] >= 0) and np.all(dataset["a_h"] < num_actions), "行動が0からnum_actions-1の範囲内であること"

    # Assert: 確率分布の検証
    for h in range(H):
        pi_0_sum = dataset["pi_0"][:, :, h].sum(axis=1)
        assert np.allclose(pi_0_sum, 1.0, atol=1e-6), f"時刻{h}におけるpi_0の総和が1.0であること"

        pi_sum = dataset["pi"][:, :, h].sum(axis=1)
        assert np.allclose(pi_sum, 1.0, atol=1e-6), f"時刻{h}におけるpiの総和が1.0であること"

    # Assert: 初期状態分布の検証
    # 初期状態s_0はinit_distに従ってサンプリングされているはず
    initial_states = dataset["s_h"][:, 0]
    assert np.all(initial_states >= 0) and np.all(initial_states < num_states), "初期状態が有効な範囲内であること"

    # Assert: 状態遷移の整合性検証
    # 各エピソードの状態遷移がtrans_probsに従っていることを確認（確率的なので完全一致はしないが、範囲は確認できる）
    for h in range(H - 1):
        current_states = dataset["s_h"][:, h]
        actions = dataset["a_h"][:, h]
        next_states = dataset["s_h"][:, h + 1]

        # 次状態が有効な範囲内であることを確認
        assert np.all(next_states >= 0) and np.all(next_states < num_states), f"時刻{h}の次状態が有効な範囲内であること"

        # trans_probsから次状態がサンプリング可能であることを確認
        for i in range(min(10, num_data)):  # サンプルで確認
            s, a, s_next = current_states[i], actions[i], next_states[i]
            # trans_probs[s, s_next, a] > 0であれば遷移可能
            # 確率0でない遷移であることを確認（厳密にはランダムなので必ずしも成り立たないが、概ね成り立つはず）
            assert trans_probs[s, s_next, a] >= 0, f"状態遷移確率が非負であること"

    # Assert: 期待報酬の整合性検証
    for h in range(H):
        for i in range(min(10, num_data)):  # サンプルで確認
            s, a = dataset["s_h"][i, h], dataset["a_h"][i, h]
            expected_q = dataset["q_s_a"][s, a]
            actual_q = dataset["q_h"][i, h]
            assert np.isclose(expected_q, actual_q, atol=1e-6), f"期待報酬q_hがq_s_aと整合していること"

    # Assert: 状態価値関数の値の範囲検証（sigmoidで正規化されているので0-1の範囲）
    assert np.all(dataset["q_s_a"] >= 0) and np.all(dataset["q_s_a"] <= 1.0), "状態価値関数が0から1の範囲内であること"

    # Assert: 再現性の検証
    dataset2 = generate_synthetic_data(
        num_data=num_data,
        dim_state=dim_state,
        num_states=num_states,
        num_actions=num_actions,
        H=H,
        theta=theta,
        M=M,
        b=b,
        init_dist=init_dist,
        trans_probs=trans_probs,
        random_state=random_state,
    )
    assert np.array_equal(dataset["s_h"], dataset2["s_h"]), "同じrandom_stateで同じ状態軌跡が生成されること"
    assert np.array_equal(dataset["a_h"], dataset2["a_h"]), "同じrandom_stateで同じ行動軌跡が生成されること"


def test_generate_synthetic_dataがis_testパラメータに応じて異なる方策を使用すること():
    # Arrange
    num_data = 50
    dim_state = 3
    num_states = 10
    num_actions = 5
    H = 3
    random_state = 42

    random_ = np.random.RandomState(random_state)
    theta = random_.normal(size=(dim_state, num_actions))
    M = random_.normal(size=(dim_state, num_actions))
    b = random_.normal(size=(1, num_actions))
    init_dist = random_.dirichlet(np.ones(num_states))
    trans_probs = random_.dirichlet(np.ones(num_states), size=(num_states, num_actions))
    trans_probs = trans_probs.transpose(0, 2, 1)

    # Act
    dataset_train = generate_synthetic_data(
        num_data=num_data,
        dim_state=dim_state,
        num_states=num_states,
        num_actions=num_actions,
        H=H,
        theta=theta,
        M=M,
        b=b,
        init_dist=init_dist,
        trans_probs=trans_probs,
        is_test=False,
        random_state=random_state,
    )

    dataset_test = generate_synthetic_data(
        num_data=num_data,
        dim_state=dim_state,
        num_states=num_states,
        num_actions=num_actions,
        H=H,
        theta=theta,
        M=M,
        b=b,
        init_dist=init_dist,
        trans_probs=trans_probs,
        is_test=True,
        random_state=random_state,
    )

    # Assert: is_testの違いにより異なるデータ収集方策が使用されること
    # データ収集方策pi_0が異なるため、軌跡も異なるはず
    assert not np.array_equal(dataset_train["pi_0"], dataset_test["pi_0"]), "is_testの違いによりpi_0が異なること"

    # 評価方策piは同じepsilon-greedyが使われる
    assert np.array_equal(dataset_train["pi"], dataset_test["pi"]), "評価方策piは同じであること"


def test_calc_true_valueが評価方策の真の性能を近似的に計算すること():
    # Arrange
    dim_state = 3
    num_states = 10
    num_actions = 5
    H = 3
    random_state = 42

    random_ = np.random.RandomState(random_state)
    theta = random_.normal(size=(dim_state, num_actions))
    M = random_.normal(size=(dim_state, num_actions))
    b = random_.normal(size=(1, num_actions))
    init_dist = random_.dirichlet(np.ones(num_states))
    trans_probs = random_.dirichlet(np.ones(num_states), size=(num_states, num_actions))
    trans_probs = trans_probs.transpose(0, 2, 1)

    # Act
    test_data, true_value = calc_true_value(
        dim_state=dim_state,
        num_states=num_states,
        num_actions=num_actions,
        H=H,
        theta=theta,
        M=M,
        b=b,
        init_dist=init_dist,
        trans_probs=trans_probs,
        num_data=1000,  # テストなので小さめ
    )

    # Assert: 返り値の検証
    assert isinstance(test_data, dict), "test_dataがdict型であること"
    assert isinstance(true_value, (float, np.floating)), "真の性能値がfloat型であること"
    assert not np.isnan(true_value), "真の性能値がNaNでないこと"
    assert not np.isinf(true_value), "真の性能値が無限大でないこと"

    # Assert: test_dataの内容検証
    assert test_data["num_data"] == 1000, "test_dataのnum_dataが指定した値であること"
    assert test_data["H"] == H, "test_dataのHが正しいこと"

    # Assert: 真の性能値の妥当性検証
    # 累積報酬の平均なので、q_s_aの範囲を考えると概ねH * 0.5程度（sigmoid後なので）
    # 厳密な範囲は難しいが、明らかに異常な値でないことを確認
    assert true_value >= 0, "真の性能値が非負であること"
    assert true_value <= H * 1.0, "真の性能値がH以下であること（各時刻の報酬が最大1なので）"

    # Assert: test_dataとtrue_valueの整合性検証
    # true_valueはq_h.sum(1).mean()で計算される
    expected_true_value = test_data["q_h"].sum(1).mean()
    assert np.isclose(true_value, expected_true_value, atol=1e-6), "真の性能値がtest_dataから計算される値と一致すること"

    # Assert: 大規模サンプルでの安定性検証
    test_data2, true_value2 = calc_true_value(
        dim_state=dim_state,
        num_states=num_states,
        num_actions=num_actions,
        H=H,
        theta=theta,
        M=M,
        b=b,
        init_dist=init_dist,
        trans_probs=trans_probs,
        num_data=1000,
    )
    # デフォルトのrandom_stateが固定されているので同じ値になる
    assert true_value == true_value2, "同じパラメータで計算した真の性能値が一致すること"
