import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy.stats import rankdata
from sklearn.utils import check_random_state


class SyntheticRankingData(BaseModel):
    """ランキングにおけるオフ方策評価用の合成データ.

    Attributes
    ----------
    num_data : int
        データ数
    K : int
        ランキングポジション数
    num_actions : int
        行動数
    x : np.ndarray
        コンテキスト特徴量 (num_data x dim_context)
    a_k : np.ndarray
        各ポジションで選択された行動 (num_data x K)
    r_k : np.ndarray
        各ポジションで観測された報酬 (num_data x K)
    C : np.ndarray
        ユーザ行動行列 (num_data x K x K)
    pi_0 : np.ndarray
        使用された方策 (num_data x num_actions)
    q_k : np.ndarray
        各ポジションの期待報酬 (num_data x K)
    base_q_func : np.ndarray
        基本Q関数の値 (num_data x num_actions)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    num_data: int
    K: int
    num_actions: int
    x: np.ndarray
    a_k: np.ndarray
    r_k: np.ndarray
    C: np.ndarray
    pi_0: np.ndarray
    q_k: np.ndarray
    base_q_func: np.ndarray


def _compute_base_q_function(
    x: np.ndarray,
    theta: np.ndarray,
    quadratic_weights: np.ndarray,
    action_bias: np.ndarray,
    num_actions: int,
) -> np.ndarray:
    """ベースとなるQ関数を計算する.

    Parameters
    ----------
    x : np.ndarray
        コンテキスト特徴量 (num_data x dim_context)
    theta : np.ndarray
        パラメータ行列 (dim_context x num_actions)
    quadratic_weights : np.ndarray
        二次項の重み行列 (dim_context x num_actions)
    action_bias : np.ndarray
        各行動のバイアス項 (num_actions x 1)
    num_actions : int
        行動数

    Returns
    -------
    np.ndarray
        Q関数の値 (num_data x num_actions)
    """
    e_a = np.eye(num_actions)
    # 非線形変換を適用してQ関数を計算
    linear_term = (x**3 + x**2 - x) @ theta
    quadratic_term = (x - x**2) @ quadratic_weights @ e_a
    return _sigmoid(linear_term + quadratic_term + action_bias.T)


def _create_user_behavior_matrices(K: int) -> np.ndarray:
    """3種類のユーザ行動パターンの行列を作成する.

    Parameters
    ----------
    K : int
        ランキングのポジション数

    Returns
    -------
    np.ndarray
        ユーザ行動パターン行列 (3 x K x K)
        - independent: 各アイテムを独立に見る
        - cascade: 順番に見る（上から下へ）
        - all: すべての組み合わせを考慮
    """
    return np.r_[
        np.eye(K),  # independent: 対角成分のみ1
        np.tril(np.ones((K, K))),  # cascade: 下三角行列
        np.ones((K, K)),  # all: すべて1
    ].reshape((3, K, K))


def _generate_random_behavior_matrices(K: int, random_: np.random.RandomState) -> np.ndarray:
    """ランダムなユーザ行動パターンの行列を生成する.

    Parameters
    ----------
    K : int
        ランキングのポジション数
    random_ : np.random.RandomState
        乱数生成器

    Returns
    -------
    np.ndarray
        ランダムユーザ行動パターン行列 (7 x K x K)
    """
    # -1, 0, 1の値をランダムに選んで7パターン生成
    return random_.choice([-1, 0, 1], p=[0.2, 0.4, 0.4], size=7 * K * K).reshape((7, K, K))


def _sample_user_behavior_matrix(
    num_data: int,
    K: int,
    p: list,
    p_rand: float,
    random_: np.random.RandomState,
) -> np.ndarray:
    """ユーザ行動行列をサンプリングする.

    Parameters
    ----------
    num_data : int
        データ数
    K : int
        ランキングのポジション数
    p : list
        各ユーザ行動パターンの選択確率 [independent, cascade, all]
    p_rand : float
        ランダム行動を追加する確率
    random_ : np.random.RandomState
        乱数生成器

    Returns
    -------
    np.ndarray
        ユーザ行動行列 C (num_data x K x K)
    """
    # 基本的なユーザ行動パターンを生成
    user_behavior_matrix = _create_user_behavior_matrices(K)
    user_behavior_idx = random_.choice(3, p=p, size=num_data)
    C_ = user_behavior_matrix[user_behavior_idx]

    # ランダムなノイズを追加
    user_behavior_matrix_rand = _generate_random_behavior_matrices(K, random_)
    user_behavior_rand_idx = random_.choice(7, size=num_data)
    C_rand = user_behavior_matrix_rand[user_behavior_rand_idx]

    # ランダム行動を追加するかどうかを決定
    is_rand = random_.binomial(2, p=p_rand, size=num_data).reshape(num_data, 1, 1)

    # 基本パターンとランダムパターンを組み合わせて[0, 1]にクリップ
    return np.clip(C_ + is_rand * C_rand, 0, 1)


def _select_policy(
    base_q_func: np.ndarray,
    beta: float,
    is_test: bool,
) -> np.ndarray:
    """方策を選択する.

    Parameters
    ----------
    base_q_func : np.ndarray
        Q関数の値
    beta : float
        温度パラメータ（softmax用）
    is_test : bool
        テストモードかどうか

    Returns
    -------
    np.ndarray
        選択された方策
    """
    if is_test:
        return _eps_greedy_policy(base_q_func)
    else:
        return _softmax(beta * base_q_func)


def _sample_actions(
    pi_0: np.ndarray,
    num_data: int,
    K: int,
    random_state: int,
) -> np.ndarray:
    """各ポジションの行動をサンプリングする.

    Parameters
    ----------
    pi_0 : np.ndarray
        方策（行動選択確率）
    num_data : int
        データ数
    K : int
        ランキングのポジション数
    random_state : int
        乱数シード

    Returns
    -------
    np.ndarray
        サンプリングされた行動 (num_data x K)
    """
    a_k = np.zeros((num_data, K), dtype=int)
    for k in range(K):
        a_k_ = _sample_action_fast(pi_0, random_state=random_state + k)
        a_k[:, k] = a_k_
    return a_k


def _compute_position_reward(
    k: int,
    idx: np.ndarray,
    a_k: np.ndarray,
    base_q_func: np.ndarray,
    C: np.ndarray,
    position_interaction_weights: np.ndarray,
    K: int,
) -> np.ndarray:
    """特定のポジションkにおける報酬を計算する.

    Parameters
    ----------
    k : int
        現在のポジション
    idx : np.ndarray
        データインデックス
    a_k : np.ndarray
        各ポジションの行動
    base_q_func : np.ndarray
        基本Q関数
    C : np.ndarray
        ユーザ行動行列
    position_interaction_weights : np.ndarray
        ポジション間の相互作用重み行列 (K x K)
    K : int
        総ポジション数

    Returns
    -------
    np.ndarray
        ポジションkの期待報酬
    """
    # 自身のポジションの基本報酬
    q_func_factual = base_q_func[idx, a_k[:, k]] / K

    # 他のポジションとの相互作用を考慮
    for position in range(K):
        if position != k:
            interaction_term = (
                C[:, k, position] * position_interaction_weights[k, position] * base_q_func[idx, a_k[:, position]]
            )
            distance_penalty = np.abs(position - k)
            q_func_factual += interaction_term / distance_penalty

    return q_func_factual


def _generate_rewards(
    num_data: int,
    K: int,
    a_k: np.ndarray,
    base_q_func: np.ndarray,
    C: np.ndarray,
    position_interaction_weights: np.ndarray,
    reward_noise: float,
    random_: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    """報酬を生成する.

    Parameters
    ----------
    num_data : int
        データ数
    K : int
        ランキングのポジション数
    a_k : np.ndarray
        各ポジションの行動
    base_q_func : np.ndarray
        基本Q関数
    C : np.ndarray
        ユーザ行動行列
    position_interaction_weights : np.ndarray
        ポジション間の相互作用重み行列 (K x K)
    reward_noise : float
        報酬ノイズの標準偏差
    random_ : np.random.RandomState
        乱数生成器

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        報酬 r_k と期待報酬 q_k
    """
    r_k = np.zeros((num_data, K), dtype=float)
    q_k = np.zeros((num_data, K), dtype=float)
    idx = np.arange(num_data)

    for k in range(K):
        q_func_factual = _compute_position_reward(k, idx, a_k, base_q_func, C, position_interaction_weights, K)
        q_k[:, k] = q_func_factual
        # ノイズを加えて実際の報酬を生成
        r_k[:, k] = random_.normal(q_func_factual, scale=reward_noise)

    return r_k, q_k


def generate_synthetic_data(
    num_data: int,
    dim_context: int,
    num_actions: int,
    K: int,
    theta: np.ndarray,  # d x |A|
    quadratic_weights: np.ndarray,  # d x |A|
    action_bias: np.ndarray,  # |A| x 1
    position_interaction_weights: np.ndarray,  # K x K
    beta: float = -1.0,
    reward_noise: float = 0.5,
    p: list = [1.0, 0.0, 0.0],  # independent, cascade, all
    p_rand: float = 0.0,
    is_test: bool = False,
    random_state: int = 12345,
) -> SyntheticRankingData:
    """ランキングにおけるオフ方策評価用の合成データを生成する.

    この関数は、コンテキスト付きランキング問題におけるユーザ行動を
    シミュレートし、ログデータを生成します。

    Parameters
    ----------
    num_data : int
        生成するデータ数
    dim_context : int
        コンテキスト特徴量の次元数
    num_actions : int
        選択可能な行動（アイテム）の総数
    K : int
        ランキングのポジション数（表示するアイテム数）
    theta : np.ndarray
        線形項のパラメータ行列 (dim_context x num_actions)
    quadratic_weights : np.ndarray
        二次項の重み行列 (dim_context x num_actions)
    action_bias : np.ndarray
        各行動のバイアス項 (num_actions x 1)
    position_interaction_weights : np.ndarray
        ポジション間の相互作用重み行列 (K x K)
    beta : float, default=-1.0
        方策のsoftmax温度パラメータ（負の値で決定的に）
    reward_noise : float, default=0.5
        報酬に加えるガウスノイズの標準偏差
    p : list, default=[1.0, 0.0, 0.0]
        ユーザ行動パターンの確率分布
        [independent, cascade, all]の選択確率
    p_rand : float, default=0.0
        ランダムな行動パターンを追加する確率
    is_test : bool, default=False
        テストモードフラグ（True: ε-greedy, False: softmax）
    random_state : int, default=12345
        乱数シード

    Returns
    -------
    SyntheticRankingData
        生成されたデータを含む構造化されたオブジェクト:
        - num_data: データ数
        - K: ランキングポジション数
        - num_actions: 行動数
        - x: コンテキスト特徴量
        - a_k: 各ポジションで選択された行動
        - r_k: 各ポジションで観測された報酬
        - C: ユーザ行動行列
        - pi_0: 使用された方策
        - q_k: 各ポジションの期待報酬
        - base_q_func: 基本Q関数の値
    """
    # 乱数生成器の初期化
    random_ = check_random_state(random_state)

    # コンテキスト特徴量の生成
    x = random_.normal(size=(num_data, dim_context))

    # 基本Q関数の計算
    base_q_func = _compute_base_q_function(x, theta, quadratic_weights, action_bias, num_actions)

    # ユーザ行動行列のサンプリング
    C = _sample_user_behavior_matrix(num_data, K, p, p_rand, random_)

    # 方策の選択
    pi_0 = _select_policy(base_q_func, beta, is_test)

    # 行動のサンプリング
    a_k = _sample_actions(pi_0, num_data, K, random_state)

    # 報酬の生成
    r_k, q_k = _generate_rewards(num_data, K, a_k, base_q_func, C, position_interaction_weights, reward_noise, random_)

    return SyntheticRankingData(
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


class RankingSyntheticBanditDataset(BaseModel):
    """ランキング問題の擬似的なバンディットデータセットを生成するクラス
    実装参考: https://github.com/ghmagazine/cfml_book/blob/main/ch2/dataset.py
    """

    dim_context: int
    num_actions: int
    k: int
    action_context: np.ndarray
    theta: np.ndarray  # 擬似的な期待報酬を生成するための設定値1つ目(d, num_actions)
    quadratic_weights: np.ndarray  # 二次項の重み行列 (d, num_actions)
    action_bias: np.ndarray  # 各行動のバイアス項 (num_actions, 1)
    position_interaction_weights: np.ndarray  # ランキング位置間の相互作用を表す重み行列 (k, k)
    beta: float = -1.0  # データ収集方策の設定値1つ目
    reward_noise: float = 0.5  # 観測報酬のばらつき度合い(標準偏差)
    p: list[float] = [0.8, 0.1, 0.1]  # ユーザ行動モデルの選択確率
    p_rand: float = 0.2  # ランダムに選択される確率
    random_state: int = 12345
    is_test: bool = False  # テストモードかどうか

    class Config:
        arbitrary_types_allowed = True  # np.ndarrayを許可

    def obtain_batch_bandit_feedback(self, num_data: int) -> SyntheticRankingData:
        """バンディットフィードバックデータを生成する.

        リファクタリングしたgenerate_synthetic_data関数を使用して、
        型安全なSyntheticRankingDataオブジェクトを返す。

        Parameters
        ----------
        num_data : int
            生成するデータ数

        Returns
        -------
        SyntheticRankingData
            生成されたランキングデータ
        """
        return generate_synthetic_data(
            num_data=num_data,
            dim_context=self.dim_context,
            num_actions=self.num_actions,
            K=self.k,
            theta=self.theta,
            quadratic_weights=self.quadratic_weights,
            action_bias=self.action_bias,
            position_interaction_weights=self.position_interaction_weights,
            beta=self.beta,
            reward_noise=self.reward_noise,
            p=self.p,
            p_rand=self.p_rand,
            is_test=self.is_test,
            random_state=self.random_state,
        )


# ヘルパー関数群
def _sigmoid(x: np.ndarray) -> np.ndarray:
    """シグモイド関数."""
    return np.exp(np.minimum(x, 0)) / (1.0 + np.exp(-np.abs(x)))


def _softmax(x: np.ndarray) -> np.ndarray:
    """ソフトマックス関数."""
    b = np.max(x, axis=1)[:, np.newaxis]
    numerator = np.exp(x - b)
    denominator = np.sum(numerator, axis=1)[:, np.newaxis]
    return numerator / denominator


def _eps_greedy_policy(
    q_func: np.ndarray,
    eps: float = 0.5,
) -> np.ndarray:
    """epsilon-greedy法により方策を定義する."""
    is_topk = rankdata(-q_func, axis=1) <= 1
    pi = (1.0 - eps) * is_topk
    pi += eps / q_func.shape[1]

    return pi


def _sample_action_fast(pi: np.ndarray, random_state: int = 12345) -> np.ndarray:
    """与えられた方策に従い、行動を高速に抽出する."""
    random_ = check_random_state(random_state)
    uniform_rvs = random_.uniform(size=pi.shape[0])[:, np.newaxis]
    cum_pi = pi.cumsum(axis=1)
    flg = cum_pi > uniform_rvs
    sampled_actions = flg.argmax(axis=1)
    return sampled_actions
