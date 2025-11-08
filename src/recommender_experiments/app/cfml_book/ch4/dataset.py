import numpy as np
from sklearn.utils import check_random_state

from recommender_experiments.app.cfml_book.common_utils import eps_greedy_policy, sample_action_fast, sigmoid, softmax


def generate_synthetic_data(
    num_data: int,
    dim_state: int,
    num_states: int,
    num_actions: int,
    H: int,
    theta: np.ndarray,  # d x |A|
    M: np.ndarray,  # d x |A|
    b: np.ndarray,  # 1 x |A|
    init_dist: np.ndarray,  # |S|
    trans_probs: np.ndarray,  # |S| x |S| x |A|
    beta: float = -1.0,
    reward_noise: float = 0.5,
    is_test: bool = False,
    random_state: int = 12345,
) -> dict:
    """強化学習におけるオフ方策評価のための合成エピソディックデータを生成する.

    有限ホライゾンH、有限状態空間|S|、有限行動空間|A|のMDPにおいて、
    エピソディックな軌跡データを生成する。

    状態価値関数q(s,a)は状態特徴量S（各状態の埋め込み表現）を用いて定義される:
        q(s,a) = sigmoid((S³ + S² - S)θ + (S - S²)M·a + b) / H
    ここでSはnum_states × dim_stateの状態埋め込み行列。

    各エピソードは以下の流れで生成される:
    1. 初期状態s_0をinit_distからサンプリング
    2. 各時刻h=0,...,H-1において:
       - データ収集方策pi_0(a|s_h)から行動a_hをサンプリング
       - 期待報酬q(s_h,a_h)に基づいてノイズ付き報酬r_hを生成
       - 状態遷移確率trans_probs[s_h,:,a_h]から次状態s_{h+1}をサンプリング

    データ収集方策pi_0は:
    - is_test=False: softmax(βq(s,a))により生成（探索的）
    - is_test=True: epsilon-greedy(k=5, eps=0.2)により生成（評価用）

    Args:
        num_data: 生成するエピソード数
        dim_state: 状態特徴量の次元数
        num_states: 状態空間のサイズ |S|
        num_actions: 行動空間のサイズ |A|
        H: エピソードのホライゾン（時間ステップ数）
        theta: 状態価値関数の線形項パラメータ (dim_state, num_actions)
        M: 状態価値関数の相互作用項パラメータ (dim_state, num_actions)
        b: 状態価値関数のバイアス項 (1, num_actions)
        init_dist: 初期状態分布 (num_states,) 確率の総和は1
        trans_probs: 状態遷移確率 (num_states, num_states, num_actions)
                     trans_probs[s, s', a] = P(s'|s,a)
        beta: softmax関数の温度パラメータ（負の値で確率分布を平滑化）
        reward_noise: 報酬のガウスノイズの標準偏差
        is_test: Trueの場合epsilon-greedy、Falseの場合softmax方策を使用
        random_state: 乱数シード

    Returns:
        以下のキーを持つエピソディックデータのdict:
            num_data: エピソード数
            H: ホライゾン
            num_states: 状態数
            num_actions: 行動数
            s_h: 状態の軌跡 (num_data, H)
            a_h: 行動の軌跡 (num_data, H)
            r_h: 報酬の軌跡 (num_data, H)
            S: 状態埋め込み行列 (num_states, dim_state)
            pi_0: データ収集方策 (num_data, num_actions, H)
            pi: 評価方策 (num_data, num_actions, H)
            q_h: 各時刻の期待報酬 (num_data, H)
            q_s_a: 状態価値関数 (num_states, num_actions)

    Examples:
        >>> random_ = np.random.RandomState(42)
        >>> num_states, num_actions, H = 10, 5, 3
        >>> dim_state = 3
        >>> theta = random_.normal(size=(dim_state, num_actions))
        >>> M = random_.normal(size=(dim_state, num_actions))
        >>> b = random_.normal(size=(1, num_actions))
        >>> init_dist = random_.dirichlet(np.ones(num_states))
        >>> trans_probs = random_.dirichlet(np.ones(num_states), size=(num_states, num_actions))
        >>> trans_probs = trans_probs.transpose(0, 2, 1)  # (S, A, S) -> (S, S, A)
        >>> dataset = generate_synthetic_data(
        ...     num_data=100,
        ...     dim_state=dim_state,
        ...     num_states=num_states,
        ...     num_actions=num_actions,
        ...     H=H,
        ...     theta=theta,
        ...     M=M,
        ...     b=b,
        ...     init_dist=init_dist,
        ...     trans_probs=trans_probs,
        ... )
        >>> dataset["s_h"].shape
        (100, 3)
        >>> dataset["r_h"].shape
        (100, 3)
    """
    random_ = check_random_state(random_state)
    S = random_.normal(size=(num_states, dim_state))
    e_a = np.eye(num_actions)
    q_s_a = sigmoid((S**3 + S**2 - S) @ theta + (S - S**2) @ M @ e_a + b) / H

    s_h = np.zeros((num_data, H), dtype=int)
    a_h = np.zeros((num_data, H), dtype=int)
    r_h = np.zeros((num_data, H))
    q_h = np.zeros((num_data, H))
    pi = np.zeros((num_data, num_actions, H))
    pi_0 = np.zeros((num_data, num_actions, H))

    s_h[:, 0] = sample_action_fast(np.tile(init_dist, (num_data, 1)), random_state=random_state)
    for h in range(H):
        if is_test:
            pi_0[:, :, h] = eps_greedy_policy(
                q_s_a[s_h[:, h]], k=5, eps=0.2, return_normalized=True, rank_method="ordinal"
            )
        else:
            pi_0[:, :, h] = softmax(beta * q_s_a[s_h[:, h]])
        pi[:, :, h] = eps_greedy_policy(q_s_a[s_h[:, h]], k=5, eps=0.2, return_normalized=True, rank_method="ordinal")
        a_h[:, h] = sample_action_fast(pi_0[:, :, h], random_state=random_state + h)
        q_h[:, h] = q_s_a[s_h[:, h], a_h[:, h]]
        r_h[:, h] = random_.normal(q_h[:, h], scale=reward_noise)
        if h < H - 1:
            s_h[:, h + 1] = sample_action_fast(trans_probs[s_h[:, h], :, a_h[:, h]], random_state=random_state + h)

    return dict(
        num_data=num_data,
        H=H,
        num_states=num_states,
        num_actions=num_actions,
        s_h=s_h,
        a_h=a_h,
        r_h=r_h,
        S=S,
        pi_0=pi_0,
        pi=pi,
        q_h=q_h,
        q_s_a=q_s_a,
    )


def calc_true_value(
    dim_state: int,
    num_states: int,
    num_actions: int,
    H: int,
    theta: np.ndarray,
    M: np.ndarray,
    b: np.ndarray,
    init_dist: np.ndarray,
    trans_probs: np.ndarray,
    num_data: int = 100000,
) -> tuple[dict, float]:
    """評価方策の真の性能を近似的に計算する.

    大規模なテストデータ(デフォルトnum_data=100000エピソード)を生成し、
    epsilon-greedy評価方策による累積報酬の平均値を計算することで、
    方策の真の性能値を近似する。

    評価方策はeps_greedy_policy(k=5, eps=0.2)により定義され、
    真の性能値はV(π) = E[Σ_{h=0}^{H-1} r_h]として計算される。

    Args:
        dim_state: 状態特徴量の次元数
        num_states: 状態空間のサイズ |S|
        num_actions: 行動空間のサイズ |A|
        H: エピソードのホライゾン
        theta: 状態価値関数の線形項パラメータ (dim_state, num_actions)
        M: 状態価値関数の相互作用項パラメータ (dim_state, num_actions)
        b: 状態価値関数のバイアス項 (1, num_actions)
        init_dist: 初期状態分布 (num_states,)
        trans_probs: 状態遷移確率 (num_states, num_states, num_actions)
        num_data: 近似計算に使用するエピソード数（デフォルト100000）

    Returns:
        (test_bandit_data, true_value)のタプル:
            test_bandit_data: 生成されたテストデータ（dict）
            true_value: 評価方策の真の性能値（エピソード累積報酬の平均値）

    Examples:
        >>> random_ = np.random.RandomState(42)
        >>> num_states, num_actions, H = 10, 5, 3
        >>> dim_state = 3
        >>> theta = random_.normal(size=(dim_state, num_actions))
        >>> M = random_.normal(size=(dim_state, num_actions))
        >>> b = random_.normal(size=(1, num_actions))
        >>> init_dist = random_.dirichlet(np.ones(num_states))
        >>> trans_probs = random_.dirichlet(np.ones(num_states), size=(num_states, num_actions))
        >>> trans_probs = trans_probs.transpose(0, 2, 1)
        >>> test_data, true_value = calc_true_value(
        ...     dim_state=dim_state,
        ...     num_states=num_states,
        ...     num_actions=num_actions,
        ...     H=H,
        ...     theta=theta,
        ...     M=M,
        ...     b=b,
        ...     init_dist=init_dist,
        ...     trans_probs=trans_probs,
        ...     num_data=1000,
        ... )
        >>> isinstance(true_value, float)
        True
        >>> isinstance(test_data, dict)
        True
    """
    test_bandit_data = generate_synthetic_data(
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
    )

    return test_bandit_data, test_bandit_data["q_h"].sum(1).mean()
