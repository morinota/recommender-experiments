import numpy as np
from sklearn.utils import check_random_state

from recommender_experiments.app.cfml_book.ch2.utils import eps_greedy_policy, sample_action_fast, sigmoid, softmax


def generate_synthetic_data(
    num_data: int,
    dim_context: int,
    num_actions: int,
    K: int,
    theta: np.ndarray,
    M: np.ndarray,
    b: np.ndarray,
    W: np.ndarray,
    beta: float = -1.0,
    reward_noise: float = 0.5,
    p: list = [1.0, 0.0, 0.0],
    p_rand: float = 0.0,
    is_test: bool = False,
    random_state: int = 12345,
) -> dict:
    """ランキング設定におけるオフ方策評価用の合成バンディットフィードバックデータを生成する.

    K個の行動をランキング形式で推薦し、ユーザがランキング内の複数アイテムを閲覧・評価する
    状況を模擬する。各位置kの報酬は基本期待報酬に加えて、他の位置の行動との相互作用効果
    （ユーザ行動行列Cと位置間重み行列Wで決定）を考慮して計算される。

    期待報酬の計算式:
        q_k = base_q(x, a_k) / K + Σ_{l≠k} C[k,l] * W[k,l] * base_q(x, a_l) / |l-k|

    ここでbase_q(x,a)はsigmoid関数で定義される基本期待報酬関数。

    Args:
        num_data: 生成するログデータのサンプル数
        dim_context: コンテキストベクトルxの次元数
        num_actions: 利用可能な行動の総数 |A|
        K: ランキングサイズ（推薦する行動の数）
        theta: 基本期待報酬関数の線形項パラメータ (dim_context, num_actions)
        M: 基本期待報酬関数の相互作用項パラメータ (dim_context, num_actions)
        b: 基本期待報酬関数のバイアス項 (1, num_actions)
        W: 位置間相互作用の重み行列 (K, K). W[k,l]は位置kから位置lへの影響度
        beta: softmax温度パラメータ（is_test=Falseの場合）. 負値でランダム性が高い
        reward_noise: 報酬に加えるガウスノイズの標準偏差
        p: ユーザ行動モデルの混合比率 [independent, cascade, all].
           - independent: 各位置を独立に閲覧（対角行列）
           - cascade: 上位から順に閲覧（下三角行列）
           - all: すべての位置を閲覧（全要素1の行列）
        p_rand: ランダムなユーザ行動パターンを混ぜる確率
        is_test: Trueの場合はeps_greedy_policyを使用、Falseの場合はsoftmax方策を使用
        random_state: 再現性のための乱数シード

    Returns:
        以下のキーを持つランキングバンディットフィードバックデータのdict:
            num_data: データ数
            K: ランキングサイズ
            num_actions: 行動数
            x: コンテキスト (num_data, dim_context)
            a_k: ランキング内の行動 (num_data, K). 各行はK個の異なる行動ID
            r_k: 各位置の報酬 (num_data, K). ガウスノイズ付き
            C: ユーザ行動行列 (num_data, K, K). C[i,k,l]=1なら位置kで位置lの行動を閲覧
            pi_0: データ収集方策の行動選択確率分布 (num_data, num_actions)
            q_k: 各位置の期待報酬 (num_data, K)
            base_q_func: 基本期待報酬関数 (num_data, num_actions)

    Examples:
        >>> random_ = np.random.RandomState(42)
        >>> theta = random_.normal(size=(5, 10))
        >>> M = random_.normal(size=(5, 10))
        >>> b = random_.normal(size=(1, 10))
        >>> W = random_.uniform(0.5, 1.5, size=(3, 3))
        >>> dataset = generate_synthetic_data(
        ...     num_data=100,
        ...     dim_context=5,
        ...     num_actions=10,
        ...     K=3,
        ...     theta=theta, M=M, b=b, W=W,
        ...     p=[0.5, 0.3, 0.2],  # 50% independent, 30% cascade, 20% all
        ... )
        >>> dataset['a_k'].shape
        (100, 3)
        >>> len(np.unique(dataset['a_k'][0]))  # ランキング内に重複なし
        3
    """
    random_ = check_random_state(random_state)
    x, e_a = random_.normal(size=(num_data, dim_context)), np.eye(num_actions)
    base_q_func = sigmoid((x**3 + x**2 - x) @ theta + (x - x**2) @ M @ e_a + b)

    # ユーザ行動モデルを抽出する
    user_behavior_matrix = np.r_[
        np.eye(K),  # independent
        np.tril(np.ones((K, K))),  # cascade
        np.ones((K, K)),  # all
    ].reshape((3, K, K))
    user_behavior_idx = random_.choice(3, p=p, size=num_data)
    C_ = user_behavior_matrix[user_behavior_idx]

    user_behavior_matrix_rand = random_.choice([-1, 0, 1], p=[0.2, 0.4, 0.4], size=7 * K * K).reshape((7, K, K))
    user_behavior_rand_idx = random_.choice(7, size=num_data)
    C_rand = user_behavior_matrix_rand[user_behavior_rand_idx]

    is_rand = random_.binomial(2, p=p_rand, size=num_data).reshape(num_data, 1, 1)
    C = np.clip(C_ + is_rand * C_rand, 0, 1)

    if is_test:
        pi_0 = eps_greedy_policy(base_q_func)
    else:
        pi_0 = softmax(beta * base_q_func)
    # 行動を抽出する
    a_k = np.zeros((num_data, K), dtype=int)
    r_k = np.zeros((num_data, K), dtype=float)
    q_k = np.zeros((num_data, K), dtype=float)
    for k in range(K):
        a_k_ = sample_action_fast(pi_0, random_state=random_state + k)
        a_k[:, k] = a_k_
    # 報酬を抽出する
    idx = np.arange(num_data)
    for k in range(K):
        q_func_factual = base_q_func[idx, a_k[:, k]] / K
        for l in range(K):
            if l != k:
                q_func_factual += C[:, k, l] * W[k, l] * base_q_func[idx, a_k[:, l]] / np.abs(l - k)
        q_k[:, k] = q_func_factual
        r_k[:, k] = random_.normal(q_func_factual, scale=reward_noise)

    return dict(
        num_data=num_data,
        K=K,
        num_actions=num_actions,
        x=x,
        a_k=a_k,
        r_k=r_k,
        C=C,
        pi_0=pi_0,
        q_k=q_k,
        base_q_func=base_q_func,
    )


def calc_true_value(
    dim_context: int,
    num_actions: int,
    K: int,
    p: list,
    theta: np.ndarray,
    M: np.ndarray,
    b: np.ndarray,
    W: np.ndarray,
    p_rand: float = 0.0,
    num_data: int = 100000,
) -> float:
    """評価方策（eps_greedy_policy）の真の性能値を大規模サンプルで近似的に計算する.

    is_test=Trueでgenerate_synthetic_data()を呼び出し、eps_greedy_policyで生成された
    大量のテストデータから期待報酬q_kの平均を計算することで、評価方策の真の性能を近似する。
    これはオフ方策評価手法の性能を評価する際のグランドトゥルースとして使用される。

    Args:
        dim_context: コンテキストベクトルxの次元数
        num_actions: 利用可能な行動の総数 |A|
        K: ランキングサイズ
        p: ユーザ行動モデルの混合比率 [independent, cascade, all]
        theta: 基本期待報酬関数の線形項パラメータ (dim_context, num_actions)
        M: 基本期待報酬関数の相互作用項パラメータ (dim_context, num_actions)
        b: 基本期待報酬関数のバイアス項 (1, num_actions)
        W: 位置間相互作用の重み行列 (K, K)
        p_rand: ランダムなユーザ行動パターンを混ぜる確率
        num_data: 近似に使用するサンプル数. デフォルト100000で十分な精度

    Returns:
        評価方策の真の性能値（ランキング全体の期待報酬の和の平均）

    Examples:
        >>> random_ = np.random.RandomState(42)
        >>> theta = random_.normal(size=(5, 10))
        >>> M = random_.normal(size=(5, 10))
        >>> b = random_.normal(size=(1, 10))
        >>> W = random_.uniform(0.5, 1.5, size=(3, 3))
        >>> true_value = calc_true_value(
        ...     dim_context=5,
        ...     num_actions=10,
        ...     K=3,
        ...     p=[0.5, 0.3, 0.2],
        ...     theta=theta, M=M, b=b, W=W,
        ...     num_data=10000,
        ... )
        >>> isinstance(true_value, float)
        True
    """
    test_bandit_data = generate_synthetic_data(
        num_data=num_data,
        dim_context=dim_context,
        num_actions=num_actions,
        K=K,
        theta=theta,
        M=M,
        b=b,
        W=W,
        is_test=True,
        p=p,
        p_rand=p_rand,
        random_state=12345,
    )

    return test_bandit_data["q_k"].sum(1).mean()
