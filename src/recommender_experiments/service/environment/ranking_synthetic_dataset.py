from typing import Optional, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict
from sklearn.utils import check_random_state

from recommender_experiments.service.algorithms.bandit_algorithm_interface import BanditAlgorithmInterface
from recommender_experiments.service.utils.math_functions import (
    eps_greedy_policy,
    sample_action_fast,
    sigmoid,
    softmax,
)


class SyntheticRankingData(BaseModel):
    """ランキングにおけるオフ方策評価用の合成データ.

    Attributes
    ----------
    num_data : int
        データ数
    ranking_positions : int
        ランキングポジション数
    num_actions : int
        行動数
    context_features : np.ndarray
        コンテキスト特徴量 (num_data x dim_context)
    selected_action_vectors : np.ndarray
        各ポジションで選択された行動 (num_data x ranking_positions)
    observed_reward_vectors : np.ndarray
        各ポジションで観測された報酬 (num_data x ranking_positions)
    user_behavior_matrix : np.ndarray
        ユーザ行動行列 (num_data x ranking_positions x ranking_positions)
    logging_policy : np.ndarray
        使用された方策 (num_data x num_actions)
    expected_rewards : np.ndarray
        各ポジションの期待報酬 (num_data x ranking_positions)
    base_q_function : np.ndarray
        基本Q関数の値 (num_data x num_actions)
    available_action_mask : np.ndarray
        各データポイントで利用可能なactionのマスク (num_data x num_actions)
        1: 利用可能, 0: 利用不可
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    num_data: int
    ranking_positions: int
    num_actions: int
    context_features: np.ndarray
    selected_action_vectors: np.ndarray
    observed_reward_vectors: np.ndarray
    user_behavior_matrix: np.ndarray
    logging_policy: np.ndarray
    expected_rewards: np.ndarray
    base_q_function: np.ndarray
    available_action_mask: np.ndarray


class RankingSyntheticBanditDataset(BaseModel):
    """このクラスは、ランキング問題におけるバンディット環境をシミュレートし、
    オフ方策評価用の合成データを生成します。

    実装参考: https://github.com/ghmagazine/cfml_book/blob/main/ch2/dataset.py

    Args:
        [環境設定]
        dim_context (int): コンテキスト特徴量の次元数
        num_actions (int): 選択可能な行動（アイテム）の総数
        k (int): ランキングで表示する上位ポジション数（推薦リストの長さ）
        action_context (np.ndarray, shape: (num_actions, action_dim)):
                      各行動のコンテキスト特徴量

        [期待報酬関数の設定値群]
        theta (np.ndarray, shape: (dim_context, num_actions)):
              Q関数の線形項に使用するパラメータ行列
              コンテキストと行動の基本的な関係性を定義
        quadratic_weights (np.ndarray, shape: (dim_context, num_actions)):
                          Q関数の二次項に使用する重み行列
                          非線形な関係性を表現するための重み
        action_bias (np.ndarray, shape: (num_actions, 1)):
                    各行動に対する固有のバイアス項
                    行動固有の選択されやすさを調整
        position_interaction_weights (np.ndarray, shape: (k, k)):
                                     ランキング位置間の相互作用重み行列
                                     ポジション間の影響度を定義

        [データ収集方策の設定値群]
        beta (float): 方策のソフトマックス温度パラメータ
                     負値で決定論的、正値で確率的な行動選択
        is_test (bool): テストモード（True: ε-greedy方策、False: softmax方策）

        [ユーザ行動モデルの設定値群]
        p (list[float], length: 3): ユーザ行動モデルの選択確率 [independent, cascade, all]
                                   independent: 各ポジションを独立して評価
                                   cascade: 上位から順次評価（途中で止める可能性）
                                   all: すべてのポジションを評価
        p_rand (float): ランダム行動パターンを混入する確率
                       ノイズのあるユーザ行動をシミュレート

        [ノイズ・再現性の設定値群]
        reward_noise (float): 観測報酬に加えるガウシアンノイズの標準偏差
                             実世界の不確実性をシミュレート
        random_state (int): 再現性確保のための乱数シード
    """

    dim_context: int
    num_actions: int
    k: int
    action_context: np.ndarray
    theta: np.ndarray
    quadratic_weights: np.ndarray
    action_bias: np.ndarray
    position_interaction_weights: np.ndarray
    beta: float = -1.0
    reward_noise: float = 0.5
    p: list[float] = [0.8, 0.1, 0.1]
    p_rand: float = 0.2
    random_state: int = 12345
    is_test: bool = False
    action_churn_schedule: Optional[dict] = None  # デフォルトではスケジュールなし

    class Config:
        arbitrary_types_allowed = True  # np.ndarrayを許可

    def obtain_batch_bandit_feedback(
        self, num_data: int, policy_algorithm: Optional[BanditAlgorithmInterface] = None
    ) -> SyntheticRankingData:
        """バンディットフィードバックデータを生成する.

        リファクタリングしたgenerate_synthetic_data関数を使用して、
        型安全なSyntheticRankingDataオブジェクトを返す。

        Parameters
        ----------
        num_data : int
            生成するデータ数
        policy_algorithm : Optional[BanditAlgorithmInterface]
            データ収集方策として使用するバンディットアルゴリズム
            Noneの場合は従来のsoftmax/ε-greedy方策を使用

        Returns
        -------
        SyntheticRankingData
            生成されたランキングデータ
        """
        # 乱数生成器の初期化
        random_ = check_random_state(self.random_state)

        # コンテキスト特徴量の生成
        x = random_.normal(size=(num_data, self.dim_context))

        # 動的action変化マスクの生成
        available_action_mask = self._generate_available_action_mask(num_data, random_)

        # 基本Q関数の計算
        base_q_func = self._compute_base_q_function(
            x, self.theta, self.quadratic_weights, self.action_bias, self.num_actions
        )

        # ユーザ行動行列のサンプリング
        C = self._sample_user_behavior_matrix(num_data, self.k, self.p, self.p_rand, random_)

        # 方策とaction選択の分岐
        if policy_algorithm is not None:
            # バンディットアルゴリズムを使用
            a_k, pi_0 = self._select_actions_with_bandit_algorithm(x, available_action_mask, policy_algorithm)
        else:
            # 従来の方策（softmax/ε-greedy）を使用
            pi_0 = self._select_policy_with_mask(base_q_func, available_action_mask, self.beta, self.is_test)
            a_k = self._sample_actions_with_mask(pi_0, available_action_mask, num_data, self.k, self.random_state)

        # 報酬の生成
        r_k, q_k = self._generate_rewards(
            num_data, self.k, a_k, base_q_func, C, self.position_interaction_weights, self.reward_noise, random_
        )

        return SyntheticRankingData(
            num_data=num_data,
            ranking_positions=self.k,
            num_actions=self.num_actions,
            context_features=x,
            selected_action_vectors=a_k,
            observed_reward_vectors=r_k,
            user_behavior_matrix=C,
            logging_policy=pi_0,
            expected_rewards=q_k,
            base_q_function=base_q_func,
            available_action_mask=available_action_mask,
        )

    def _compute_base_q_function(
        self,
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
        return sigmoid(linear_term + quadratic_term + action_bias.T)

    def _create_user_behavior_matrices(self, K: int) -> np.ndarray:
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

    def _generate_random_behavior_matrices(self, K: int, random_: np.random.RandomState) -> np.ndarray:
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
        self,
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
        user_behavior_matrix = self._create_user_behavior_matrices(K)
        user_behavior_idx = random_.choice(3, p=p, size=num_data)
        C_ = user_behavior_matrix[user_behavior_idx]

        # ランダムなノイズを追加
        user_behavior_matrix_rand = self._generate_random_behavior_matrices(K, random_)
        user_behavior_rand_idx = random_.choice(7, size=num_data)
        C_rand = user_behavior_matrix_rand[user_behavior_rand_idx]

        # ランダム行動を追加するかどうかを決定
        is_rand = random_.binomial(2, p=p_rand, size=num_data).reshape(num_data, 1, 1)

        # 基本パターンとランダムパターンを組み合わせて[0, 1]にクリップ
        return np.clip(C_ + is_rand * C_rand, 0, 1)

    def _select_policy(
        self,
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
            return eps_greedy_policy(base_q_func)
        else:
            return softmax(beta * base_q_func)

    def _sample_actions(
        self,
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
            a_k_ = sample_action_fast(pi_0, random_state=random_state + k)
            a_k[:, k] = a_k_
        return a_k

    def _compute_position_reward(
        self,
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
        self,
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
            q_func_factual = self._compute_position_reward(k, idx, a_k, base_q_func, C, position_interaction_weights, K)
            q_k[:, k] = q_func_factual
            # ノイズを加えて実際の報酬を生成
            r_k[:, k] = random_.normal(q_func_factual, scale=reward_noise)

        return r_k, q_k

    def _generate_available_action_mask(self, num_data: int, random_: np.random.RandomState) -> np.ndarray:
        """動的action変化に基づいて利用可能actionマスクを生成する.

        Parameters
        ----------
        num_data : int
            データ数
        random_ : np.random.RandomState
            乱数生成器

        Returns
        -------
        np.ndarray
            利用可能actionマスク (num_data x num_actions)
            1: 利用可能, 0: 利用不可
        """
        mask = np.ones((num_data, self.num_actions), dtype=int)

        # action_churn_scheduleが指定されている場合
        if self.action_churn_schedule is not None:
            mask = np.zeros((num_data, self.num_actions), dtype=int)

            # 各データポイントに対して、該当するスケジュールからactionを決定
            for i in range(num_data):
                # 現在のデータポイントに適用されるスケジュールを探す
                applicable_actions = None
                for start_idx in sorted(self.action_churn_schedule.keys(), reverse=True):
                    if i >= start_idx:
                        applicable_actions = self.action_churn_schedule[start_idx]
                        break

                if applicable_actions is not None:
                    mask[i, applicable_actions] = 1
                else:
                    # スケジュールが見つからない場合はデフォルトで全action利用可能
                    mask[i, :] = 1

        # 最低1つのactionは利用可能にする（ゼロ除算回避）
        for i in range(num_data):
            if np.sum(mask[i]) == 0:
                mask[i, random_.randint(0, self.num_actions)] = 1

        return mask

    def _select_policy_with_mask(
        self, base_q_func: np.ndarray, available_action_mask: np.ndarray, beta: float, is_test: bool
    ) -> np.ndarray:
        """利用可能actionマスクを考慮した方策を選択する.

        Parameters
        ----------
        base_q_func : np.ndarray
            Q関数の値 (num_data x num_actions)
        available_action_mask : np.ndarray
            利用可能actionマスク (num_data x num_actions)
        beta : float
            温度パラメータ（softmax用）
        is_test : bool
            テストモードかどうか

        Returns
        -------
        np.ndarray
            選択された方策 (num_data x num_actions)
        """
        num_data, num_actions = base_q_func.shape
        policy = np.zeros((num_data, num_actions))

        for i in range(num_data):
            # 利用可能actionのみを取得
            available_actions = np.where(available_action_mask[i] == 1)[0]
            available_q_values = base_q_func[i, available_actions]

            if is_test:
                # ε-greedy方策（利用可能actionのみで）
                policy_available = eps_greedy_policy(available_q_values.reshape(1, -1))[0]
            else:
                # ソフトマックス方策（利用可能actionのみで）
                policy_available = softmax(beta * available_q_values.reshape(1, -1))[0]

            # 利用可能actionに方策を割り当て
            policy[i, available_actions] = policy_available

        return policy

    def _select_actions_with_bandit_algorithm(
        self,
        x: np.ndarray,
        available_action_mask: np.ndarray,
        policy_algorithm: BanditAlgorithmInterface,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """バンディットアルゴリズムを使用してactionを選択する.

        Parameters
        ----------
        x : np.ndarray
            コンテキスト特徴量 (num_data x dim_context)
        available_action_mask : np.ndarray
            利用可能actionマスク (num_data x num_actions)
        policy_algorithm : BanditAlgorithmInterface
            使用するバンディットアルゴリズム

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (selected_actions, logging_policy) のタプル
            selected_actions: 選択されたaction (num_data x k)
            logging_policy: ログ方策 (num_data x num_actions)
        """
        num_data, num_actions = available_action_mask.shape
        selected_actions = np.zeros((num_data, self.k), dtype=int)
        logging_policy = np.zeros((num_data, num_actions))

        for i in range(num_data):
            # 利用可能actionを取得
            available_actions = np.where(available_action_mask[i] == 1)[0]

            if len(available_actions) == 0:
                continue

            # バンディットアルゴリズムでaction選択（学習しない固定状態）
            selected_action_list = policy_algorithm.select_actions(
                context=x[i], available_actions=available_actions, k=self.k
            )

            # 選択されたactionを記録
            for j, action_id in enumerate(selected_action_list):
                if j < self.k:
                    selected_actions[i, j] = action_id

            # ログ方策を計算（選択されたactionに均等な確率を割り当て）
            # 注意: 実際のバンディットアルゴリズムは方策確率を直接提供しないため、
            # 選択されたactionに基づいて近似的な方策を構築
            selected_count = min(len(selected_action_list), self.k)
            if selected_count > 0:
                prob_per_selected = 1.0 / selected_count
                for j in range(selected_count):
                    if j < len(selected_action_list):
                        action_id = selected_action_list[j]
                        logging_policy[i, action_id] = prob_per_selected

        return selected_actions, logging_policy

    def _sample_actions_with_mask(
        self,
        pi_0: np.ndarray,
        available_action_mask: np.ndarray,
        num_data: int,
        K: int,
        random_state: int,
    ) -> np.ndarray:
        """利用可能actionマスクを考慮した行動サンプリング.

        Parameters
        ----------
        pi_0 : np.ndarray
            方策（行動選択確率） (num_data x num_actions)
        available_action_mask : np.ndarray
            利用可能actionマスク (num_data x num_actions)
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
        random_ = check_random_state(random_state)

        for i in range(num_data):
            # 利用可能actionのみから方策を作成
            available_actions = np.where(available_action_mask[i] == 1)[0]
            available_probs = pi_0[i, available_actions]

            # 確率の正規化（念のため）
            if np.sum(available_probs) > 0:
                available_probs = available_probs / np.sum(available_probs)
            else:
                # 全ての確率が0の場合は均等分布
                available_probs = np.ones(len(available_actions)) / len(available_actions)

            # K回のサンプリング（重複あり）
            for k in range(K):
                selected_idx = random_.choice(len(available_actions), p=available_probs)
                a_k[i, k] = available_actions[selected_idx]

        return a_k
