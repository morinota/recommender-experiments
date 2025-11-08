import numpy as np
from sklearn.utils import check_random_state

from recommender_experiments.app.cfml_book.common_utils import eps_greedy_policy, sample_action_fast, softmax


def generate_synthetic_data(
    num_data: int,
    theta_g: np.ndarray,
    M_g: np.ndarray,
    b_g: np.ndarray,
    theta_h: np.ndarray,
    M_h: np.ndarray,
    b_h: np.ndarray,
    phi_a: np.ndarray,
    lambda_: float = 0.5,
    dim_context: int = 5,
    num_actions: int = 50,
    num_def_actions: int = 0,
    num_clusters: int = 3,
    beta: float = -3.0,
    random_state: int = 12345,
) -> dict:
    """行動特徴量を活用したオフ方策評価におけるクラスタ構造を持つ合成バンディットフィードバックデータを生成する.

    期待報酬関数はクラスタ効果g(x,c)と残差効果h(x,a)の重み付き和として定義される:
        q(x,a) = (1-λ)g(x,c) + λh(x,a)
    ここでcはphi_a[a]により行動aにマッピングされたクラスタ。

    クラスタ効果g(x,c)と残差効果h(x,a)はそれぞれ異なる非線形変換を用いて定義される:
        g(x,c) = ((x - x²)θ_g + (x³ + x² - x)M_g・c + b_g) / 10
        h(x,a) = ((x³ + x² - x)θ_h + (x - x²)M_h・a + b_h) / 10

    データ収集方策pi_0はsoftmax(βq(x,a))により生成される。
    deficient support設定の場合、最初のnum_def_actions個の行動には確率0が割り当てられる。

    Args:
        num_data: 生成するログデータのサンプル数
        theta_g: クラスタ効果g(x,c)の線形項パラメータ (dim_context, num_clusters)
        M_g: クラスタ効果g(x,c)の相互作用項パラメータ (dim_context, num_clusters)
        b_g: クラスタ効果g(x,c)のバイアス項 (1, num_clusters)
        theta_h: 残差効果h(x,a)の線形項パラメータ (dim_context, num_actions)
        M_h: 残差効果h(x,a)の相互作用項パラメータ (dim_context, num_actions)
        b_h: 残差効果h(x,a)のバイアス項 (1, num_actions)
        phi_a: 行動からクラスタへのマッピング (num_actions,) 各要素は0からnum_clusters-1の整数
        lambda_: クラスタ効果と残差効果の重み付けパラメータ (0から1の範囲)
        dim_context: コンテキスト特徴量の次元数
        num_actions: 行動数
        num_def_actions: deficient support設定で確率0を割り当てる行動数
        num_clusters: クラスタ数
        beta: softmax関数の温度パラメータ（負の値で確率分布を平滑化）
        random_state: 乱数シード

    Returns:
        以下のキーを持つバンディットフィードバックデータのdict:
            num_data: データ数
            num_actions: 行動数
            num_clusters: クラスタ数
            x: コンテキスト特徴量 (num_data, dim_context)
            a: 選択された行動 (num_data,)
            c: 選択された行動に対応するクラスタ (num_data,)
            r: 観測された報酬 (num_data,)
            phi_a: 行動からクラスタへのマッピング (num_actions,)
            pi_0: データ収集方策の行動選択確率 (num_data, num_actions)
            g_x_c: クラスタ効果成分 (1-λ)g(x,c) (num_data, num_clusters)
            h_x_a: 残差効果成分 λh(x,a) (num_data, num_actions)
            q_x_a: 期待報酬関数 (num_data, num_actions)

    Examples:
        >>> random_ = np.random.RandomState(42)
        >>> num_actions, num_clusters = 50, 3
        >>> dim_context = 5
        >>> phi_a = random_.choice(num_clusters, size=num_actions)
        >>> theta_g = random_.normal(size=(dim_context, num_clusters))
        >>> M_g = random_.normal(size=(dim_context, num_clusters))
        >>> b_g = random_.normal(size=(1, num_clusters))
        >>> theta_h = random_.normal(size=(dim_context, num_actions))
        >>> M_h = random_.normal(size=(dim_context, num_actions))
        >>> b_h = random_.normal(size=(1, num_actions))
        >>> dataset = generate_synthetic_data(
        ...     num_data=100,
        ...     theta_g=theta_g,
        ...     M_g=M_g,
        ...     b_g=b_g,
        ...     theta_h=theta_h,
        ...     M_h=M_h,
        ...     b_h=b_h,
        ...     phi_a=phi_a,
        ...     lambda_=0.5,
        ... )
        >>> dataset["x"].shape
        (100, 5)
        >>> dataset["q_x_a"].shape
        (100, 50)
    """
    random_ = check_random_state(random_state)
    x = random_.normal(size=(num_data, dim_context))
    one_hot_a, one_hot_c = np.eye(num_actions), np.eye(num_clusters)

    # 期待報酬関数を定義する
    g_x_c = ((x - x**2) @ theta_g + (x**3 + x**2 - x) @ M_g @ one_hot_c + b_g) / 10
    h_x_a = ((x**3 + x**2 - x) @ theta_h + (x - x**2) @ M_h @ one_hot_a + b_h) / 10
    q_x_a = (1 - lambda_) * g_x_c[:, phi_a] + lambda_ * h_x_a

    # データ収集方策を定義する
    pi_0 = softmax(beta * q_x_a)
    pi_0[:, :num_def_actions] = 0
    pi_0 = pi_0 / pi_0.sum(1)[:, np.newaxis]

    # 行動や報酬を抽出する
    a = sample_action_fast(pi_0, random_state=random_state)
    q_x_a_factual = q_x_a[np.arange(num_data), a]
    r = random_.normal(q_x_a_factual)

    return dict(
        num_data=num_data,
        num_actions=num_actions,
        num_clusters=num_clusters,
        x=x,
        a=a,
        c=phi_a[a],
        r=r,
        phi_a=phi_a,
        pi_0=pi_0,
        g_x_c=(1 - lambda_) * g_x_c,
        h_x_a=lambda_ * h_x_a,
        q_x_a=q_x_a,
    )


def calc_true_value(
    dim_context: int,
    num_actions: int,
    num_clusters: int,
    theta_g: np.ndarray,
    M_g: np.ndarray,
    b_g: np.ndarray,
    phi_a: np.ndarray,
    theta_h: np.ndarray,
    M_h: np.ndarray,
    b_h: np.ndarray,
    lambda_: float,
) -> float:
    """評価方策の真の性能を近似的に計算する.

    大規模なテストデータ(num_data=10000)を生成し、epsilon-greedy方策による
    期待報酬の平均値を計算することで、方策の真の性能値を近似する。

    評価方策はeps_greedy_policy(k=5, eps=0.1)により定義され、
    真の性能値はV(π) = E_x[Σ_a π(a|x)q(x,a)]として計算される。

    Args:
        dim_context: コンテキスト特徴量の次元数
        num_actions: 行動数
        num_clusters: クラスタ数
        theta_g: クラスタ効果g(x,c)の線形項パラメータ (dim_context, num_clusters)
        M_g: クラスタ効果g(x,c)の相互作用項パラメータ (dim_context, num_clusters)
        b_g: クラスタ効果g(x,c)のバイアス項 (1, num_clusters)
        phi_a: 行動からクラスタへのマッピング (num_actions,)
        theta_h: 残差効果h(x,a)の線形項パラメータ (dim_context, num_actions)
        M_h: 残差効果h(x,a)の相互作用項パラメータ (dim_context, num_actions)
        b_h: 残差効果h(x,a)のバイアス項 (1, num_actions)
        lambda_: クラスタ効果と残差効果の重み付けパラメータ

    Returns:
        評価方策の真の性能値（期待報酬の平均値）

    Examples:
        >>> random_ = np.random.RandomState(42)
        >>> num_actions, num_clusters, dim_context = 50, 3, 5
        >>> phi_a = random_.choice(num_clusters, size=num_actions)
        >>> theta_g = random_.normal(size=(dim_context, num_clusters))
        >>> M_g = random_.normal(size=(dim_context, num_clusters))
        >>> b_g = random_.normal(size=(1, num_clusters))
        >>> theta_h = random_.normal(size=(dim_context, num_actions))
        >>> M_h = random_.normal(size=(dim_context, num_actions))
        >>> b_h = random_.normal(size=(1, num_actions))
        >>> true_value = calc_true_value(
        ...     dim_context=dim_context,
        ...     num_actions=num_actions,
        ...     num_clusters=num_clusters,
        ...     theta_g=theta_g,
        ...     M_g=M_g,
        ...     b_g=b_g,
        ...     phi_a=phi_a,
        ...     theta_h=theta_h,
        ...     M_h=M_h,
        ...     b_h=b_h,
        ...     lambda_=0.5,
        ... )
        >>> isinstance(true_value, float)
        True
    """
    test_bandit_data = generate_synthetic_data(
        num_data=10000,
        dim_context=dim_context,
        num_actions=num_actions,
        num_clusters=num_clusters,
        theta_g=theta_g,
        M_g=M_g,
        b_g=b_g,
        theta_h=theta_h,
        M_h=M_h,
        b_h=b_h,
        lambda_=lambda_,
        phi_a=phi_a,
    )

    q_x_a = test_bandit_data["q_x_a"]
    pi = eps_greedy_policy(q_x_a, k=5, eps=0.1, return_normalized=True, rank_method="ordinal")

    return (q_x_a * pi).sum(1).mean()
