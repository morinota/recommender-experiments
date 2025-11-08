import numpy as np
from sklearn.utils import check_random_state

from recommender_experiments.app.cfml_book.ch5.utils import logging_policy, sample_action_fast, sigmoid


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
    num_clusters: int = 3,
    beta: float = 1.0,
    lam: float = 0.5,
    sigma: float = 1.0,
    random_state: int = 12345,
) -> dict:
    """オフ方策学習におけるクラスタ構造を持つ合成バンディットフィードバックデータを生成する.

    期待報酬関数はクラスタ効果g(x,c)と残差効果h(x,a)の重み付き和として定義される:
        q(x,a) = (1-λ)g(x,c) + λh(x,a)
    ここでcはphi_a[a]により行動aにマッピングされたクラスタ。

    データ収集方策pi_0はlogging_policy()により期待報酬をベースに生成され、
    その方策に従って行動が選択され、選択された行動の期待報酬に基づいて
    バイナリ報酬が確率的に生成される。

    Args:
        num_data: 生成するログデータのサンプル数
        theta_g: クラスタ効果g(x,c)の線形項パラメータ (dim_context, num_clusters)
        M_g: クラスタ効果g(x,c)の相互作用項パラメータ (dim_context, num_clusters)
        b_g: クラスタ効果g(x,c)のバイアス項 (1, num_clusters)
        theta_h: 残差効果h(x,a)の線形項パラメータ (dim_context, num_actions)
        M_h: 残差効果h(x,a)の相互作用項パラメータ (dim_context, num_actions)
        b_h: 残差効果h(x,a)のバイアス項 (1, num_actions)
        phi_a: 各行動のクラスタIDマッピング (num_actions,). 各要素は0からnum_clusters-1の整数
        lambda_: クラスタ効果と残差効果の配合率. 0=クラスタのみ, 1=残差のみ
        dim_context: コンテキストベクトルxの次元数
        num_actions: 行動数 |A|
        num_clusters: クラスタ数 |C|
        beta: logging_policyのsoftmax温度パラメータ
        lam: logging_policyの期待報酬とノイズの配合率
        sigma: logging_policyに加えるノイズの標準偏差
        random_state: 再現性のための乱数シード

    Returns:
        以下のキーを持つバンディットフィードバックデータのdict:
            num_data: データ数
            num_actions: 行動数
            num_clusters: クラスタ数
            x: コンテキスト (num_data, dim_context)
            a: 選択された行動 (num_data,)
            c: 選択された行動のクラスタID (num_data,)
            r: 観測された報酬 (num_data,). 0または1のバイナリ値
            phi_a: 行動とクラスタのマッピング (num_actions,)
            pi_0: データ収集方策の行動選択確率分布 (num_data, num_actions)
            pi_0_c: データ収集方策のクラスタ選択確率分布 (num_data, num_clusters)
            pscore: 選択された行動の傾向スコア (num_data,)
            pscore_c: 選択された行動のクラスタ傾向スコア (num_data,)
            g_x_c: 重み付き後のクラスタ効果 (1-λ)g(x,c) (num_data, num_clusters)
            h_x_a: 重み付き後の残差効果 λh(x,a) (num_data, num_actions)
            q_x_a: 期待報酬関数 q(x,a) (num_data, num_actions)

    Examples:
        >>> random_ = np.random.RandomState(42)
        >>> phi_a = random_.choice(3, size=10)  # 10行動を3クラスタにマッピング
        >>> theta_g = random_.normal(size=(5, 3))
        >>> M_g = random_.normal(size=(5, 3))
        >>> b_g = random_.normal(size=(1, 3))
        >>> theta_h = random_.normal(size=(5, 10))
        >>> M_h = random_.normal(size=(5, 10))
        >>> b_h = random_.normal(size=(1, 10))
        >>> dataset = generate_synthetic_data(
        ...     num_data=100,
        ...     theta_g=theta_g, M_g=M_g, b_g=b_g,
        ...     theta_h=theta_h, M_h=M_h, b_h=b_h,
        ...     phi_a=phi_a,
        ...     dim_context=5,
        ...     num_actions=10,
        ...     num_clusters=3,
        ... )
        >>> dataset['x'].shape
        (100, 5)
        >>> dataset['a'].shape
        (100,)
        >>> np.all(np.isin(dataset['r'], [0, 1]))
        True
    """
    random_ = check_random_state(random_state)
    x = random_.normal(size=(num_data, dim_context))
    one_hot_a, one_hot_c = np.eye(num_actions), np.eye(num_clusters)

    # 期待報酬関数を定義する
    g_x_c = sigmoid((x - x**2) @ theta_g + (x**3 + x**2 - x) @ M_g @ one_hot_c + b_g)
    h_x_a = sigmoid((x**3 + x**2 - x) @ theta_h + (x - x**2) @ M_h @ one_hot_a + b_h)
    q_x_a = (1 - lambda_) * g_x_c[:, phi_a] + lambda_ * h_x_a

    # データ収集方策を定義する
    pi_0 = logging_policy(q_x_a, beta=beta, sigma=sigma, lam=lam)
    idx = np.arange(num_data)
    pi_0_c = np.zeros((num_data, num_clusters))
    for c_ in range(num_clusters):
        pi_0_c[:, c_] = pi_0[:, phi_a == c_].sum(1)

    # 行動や報酬を抽出する
    a = sample_action_fast(pi_0, random_state=random_state)
    q_x_a_factual = q_x_a[idx, a]
    r = random_.binomial(n=1, p=q_x_a_factual)

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
        pi_0_c=pi_0_c,
        pscore=pi_0[idx, a],
        pscore_c=pi_0_c[idx, phi_a[a]],
        g_x_c=(1 - lambda_) * g_x_c,
        h_x_a=lambda_ * h_x_a,
        q_x_a=q_x_a,
    )
