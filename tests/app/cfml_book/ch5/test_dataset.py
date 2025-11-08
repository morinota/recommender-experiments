import numpy as np
import pytest

from recommender_experiments.app.cfml_book.ch5.dataset import generate_synthetic_data


@pytest.fixture
def basic_params():
    """基本的なパラメータセット（テスト用に軽量化）"""
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

    return {
        "num_data": num_data,
        "dim_context": dim_context,
        "num_actions": num_actions,
        "num_clusters": num_clusters,
        "phi_a": phi_a,
        "theta_g": theta_g,
        "M_g": M_g,
        "b_g": b_g,
        "theta_h": theta_h,
        "M_h": M_h,
        "b_h": b_h,
        "random_state": random_state,
    }


def test_生成されるログデータが必須フィールドをすべて含むこと(basic_params):
    # Arrange
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

    # Act
    dataset = generate_synthetic_data(**basic_params)

    # Assert
    for key in required_keys:
        assert key in dataset, f"{key}がログデータに含まれること"


def test_コンテキストの形状がnum_data掛けるdim_contextであること(basic_params):
    # Arrange & Act
    dataset = generate_synthetic_data(**basic_params)

    # Assert
    expected_shape = (basic_params["num_data"], basic_params["dim_context"])
    assert dataset["x"].shape == expected_shape, f"コンテキストxの形状が{expected_shape}であること"


def test_選択された行動がnum_actionsの範囲内であること(basic_params):
    # Arrange & Act
    dataset = generate_synthetic_data(**basic_params)

    # Assert
    assert len(dataset["a"]) == basic_params["num_data"], "行動aの数がnum_dataと一致すること"
    assert np.all(dataset["a"] >= 0), "すべての行動が0以上であること"
    assert np.all(dataset["a"] < basic_params["num_actions"]), "すべての行動がnum_actions未満であること"


def test_報酬が0または1のバイナリ値であること(basic_params):
    # Arrange & Act
    dataset = generate_synthetic_data(**basic_params)

    # Assert
    assert len(dataset["r"]) == basic_params["num_data"], "報酬rの数がnum_dataと一致すること"
    assert np.all(np.isin(dataset["r"], [0, 1])), "報酬rが0または1のバイナリ値であること"


def test_行動選択確率の総和が1になること(basic_params):
    # Arrange & Act
    dataset = generate_synthetic_data(**basic_params)

    # Assert
    pi_0_sum = dataset["pi_0"].sum(axis=1)
    assert np.allclose(pi_0_sum, 1.0, atol=1e-6), "各コンテキストにおけるpi_0の総和が1.0であること"


def test_クラスタ選択確率の総和が1になること(basic_params):
    # Arrange & Act
    dataset = generate_synthetic_data(**basic_params)

    # Assert
    pi_0_c_sum = dataset["pi_0_c"].sum(axis=1)
    assert np.allclose(pi_0_c_sum, 1.0, atol=1e-6), "各コンテキストにおけるpi_0_cの総和が1.0であること"


def test_pscoreが選択された行動の行動選択確率と一致すること(basic_params):
    # Arrange & Act
    dataset = generate_synthetic_data(**basic_params)

    # Assert
    idx = np.arange(basic_params["num_data"])
    expected_pscore = dataset["pi_0"][idx, dataset["a"]]
    assert np.allclose(dataset["pscore"], expected_pscore), "pscoreが選択された行動aのpi_0と一致すること"


def test_期待報酬q_x_aがg_x_cとh_x_aの重み付き和であること(basic_params):
    # Arrange
    lambda_ = 0.6

    # Act
    dataset = generate_synthetic_data(**basic_params, lambda_=lambda_)

    # Assert
    # g_x_cとh_x_aはすでに係数が掛けられている
    # g_x_c = (1-lambda_) * g_x_c_raw
    # h_x_a = lambda_ * h_x_a_raw
    # q_x_a = g_x_c_raw + h_x_a_raw (phi_aによるマッピング込み)
    phi_a = basic_params["phi_a"]
    g_contribution = dataset["g_x_c"][:, phi_a]  # 各行動に対応するクラスタのg値
    h_contribution = dataset["h_x_a"]
    reconstructed_q = g_contribution + h_contribution
    assert np.allclose(dataset["q_x_a"], reconstructed_q, atol=1e-10), "q_x_aがg_x_cとh_x_aの和であること"


def test_期待報酬が0から1の範囲内であること(basic_params):
    # Arrange & Act
    dataset = generate_synthetic_data(**basic_params)

    # Assert
    assert np.all(dataset["q_x_a"] >= 0), "期待報酬q_x_aが0以上であること"
    assert np.all(dataset["q_x_a"] <= 1), "期待報酬q_x_aが1以下であること"


def test_同じrandom_stateで再現可能な結果が得られること(basic_params):
    # Arrange & Act
    dataset1 = generate_synthetic_data(**basic_params)
    dataset2 = generate_synthetic_data(**basic_params)

    # Assert
    assert np.array_equal(dataset1["x"], dataset2["x"]), "同じrandom_stateでコンテキストxが再現されること"
    assert np.array_equal(dataset1["a"], dataset2["a"]), "同じrandom_stateで行動aが再現されること"
    assert np.array_equal(dataset1["r"], dataset2["r"]), "同じrandom_stateで報酬rが再現されること"


def test_lambda値が期待報酬の構成比に影響すること(basic_params):
    # Arrange
    lambda_0 = 0.0  # クラスタ効果のみ
    lambda_1 = 1.0  # 残差効果のみ

    # Act
    dataset_cluster_only = generate_synthetic_data(**basic_params, lambda_=lambda_0)
    dataset_residual_only = generate_synthetic_data(**basic_params, lambda_=lambda_1)

    # Assert
    # lambda=0の場合、h_x_a（残差）の寄与が0になる
    assert np.allclose(dataset_cluster_only["h_x_a"], 0.0, atol=1e-10), "lambda=0の時h_x_aが0であること"

    # lambda=1の場合、g_x_c（クラスタ）の寄与が0になる
    assert np.allclose(dataset_residual_only["g_x_c"], 0.0, atol=1e-10), "lambda=1の時g_x_cが0であること"


def test_クラスタ数と行動数の関係が正しく反映されること(basic_params):
    # Arrange & Act
    dataset = generate_synthetic_data(**basic_params)

    # Assert
    assert len(dataset["phi_a"]) == basic_params["num_actions"], "phi_aの長さがnum_actionsと一致すること"
    assert np.all(dataset["phi_a"] >= 0), "すべてのクラスタIDが0以上であること"
    assert np.all(
        dataset["phi_a"] < basic_params["num_clusters"]
    ), "すべてのクラスタIDがnum_clusters未満であること"

    # 選択された行動のクラスタが正しくマッピングされている
    expected_c = dataset["phi_a"][dataset["a"]]
    assert np.array_equal(dataset["c"], expected_c), "cがphi_a[a]と一致すること"
