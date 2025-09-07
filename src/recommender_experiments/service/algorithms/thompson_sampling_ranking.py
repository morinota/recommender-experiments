"""Thompson Samplingを使ったランキングバンディットアルゴリズム."""

import numpy as np
from sklearn.utils import check_random_state

from recommender_experiments.service.algorithms.bandit_algorithm_interface import BanditAlgorithmInterface


class ThompsonSamplingRanking(BanditAlgorithmInterface):
    """Thompson Samplingを使ったランキングバンディット.

    線形リワードモデルを仮定し、各actionの報酬をGaussian分布で近似。
    ベイズ線形回帰を使用してパラメータの事後分布を更新し、
    サンプリングによって行動を選択する。

    Parameters
    ----------
    num_actions : int
        行動数
    k : int
        ランキング長（選択する行動数）
    dim_context : int
        コンテキスト特徴量の次元数
    alpha : float
        事前分布の精度パラメータ（正則化項）
    beta : float
        観測ノイズの精度パラメータ
    random_state : int
        乱数シード
    """

    def __init__(
        self,
        num_actions: int,
        k: int,
        dim_context: int,
        alpha: float = 1.0,
        beta: float = 1.0,
        random_state: int = 42,
    ):
        self.num_actions = num_actions
        self.k = k
        self.dim_context = dim_context
        self.alpha = alpha
        self.beta = beta
        self.random_state = random_state
        self.random_ = check_random_state(random_state)

        # ベイズ線形回帰のパラメータを初期化
        self.reset()

    def reset(self) -> None:
        """アルゴリズムの状態をリセットする."""
        # 各actionに対する事後分布のパラメータ
        # S_a = alpha * I + beta * sum(x_t * x_t^T)  (精度行列)
        # m_a = beta * S_a^{-1} * sum(x_t * r_t)     (平均ベクトル)
        self.S = np.array([self.alpha * np.eye(self.dim_context) for _ in range(self.num_actions)])
        self.m = np.zeros((self.num_actions, self.dim_context))

        # 効率化のために逆行列もキャッシュ
        self.S_inv = np.array([np.eye(self.dim_context) / self.alpha for _ in range(self.num_actions)])

    def select_actions(self, context: np.ndarray, available_actions: np.ndarray, k: int) -> list[int]:
        """利用可能なactionの中からk個の行動を選択する.

        Thompson Samplingにより、各actionの報酬期待値を事後分布から
        サンプリングし、上位k個を選択する。

        Parameters
        ----------
        context : np.ndarray
            コンテキスト特徴量 (dim_context,)
        available_actions : np.ndarray
            利用可能なaction IDの配列
        k : int
            選択する行動数

        Returns
        -------
        List[int]
            選択された行動のリスト（長さk）
        """
        if len(available_actions) == 0:
            return []

        # 利用可能な行動数がk未満の場合は全て選択
        k = min(k, len(available_actions))

        sampled_rewards = []

        for action_id in available_actions:
            # 事後分布からパラメータをサンプリング
            # theta ~ N(m_a, S_a^{-1} / beta)
            cov = self.S_inv[action_id] / self.beta
            theta_sample = self.random_.multivariate_normal(self.m[action_id], cov)

            # サンプリングされたパラメータで報酬期待値を計算
            expected_reward = np.dot(context, theta_sample)
            sampled_rewards.append((expected_reward, action_id))

        # 報酬期待値の降順でソートしてtop-kを選択
        sampled_rewards.sort(key=lambda x: x[0], reverse=True)
        selected_actions = [action_id for _, action_id in sampled_rewards[:k]]

        return selected_actions

    def update(self, context: np.ndarray, selected_actions: list[int], rewards: list[float]) -> None:
        """観測された報酬をもとにベイズ線形回帰のパラメータを更新する.

        Parameters
        ----------
        context : np.ndarray
            コンテキスト特徴量 (dim_context,)
        selected_actions : List[int]
            選択された行動のリスト
        rewards : List[float]
            各行動に対応する報酬のリスト
        """
        for action_id, reward in zip(selected_actions, rewards):
            if action_id >= self.num_actions:
                continue  # 無効なaction IDはスキップ

            # 精度行列を更新: S_a = S_a + beta * x * x^T
            self.S[action_id] += self.beta * np.outer(context, context)

            # 逆行列を効率的に更新 (Sherman-Morrison formula)
            # (A + uv^T)^{-1} = A^{-1} - (A^{-1}uv^T A^{-1}) / (1 + v^T A^{-1} u)
            A_inv = self.S_inv[action_id]
            u = v = context * np.sqrt(self.beta)

            denominator = 1.0 + np.dot(v, np.dot(A_inv, u))
            if abs(denominator) > 1e-10:  # 数値安定性のチェック
                numerator = np.outer(np.dot(A_inv, u), np.dot(v, A_inv))
                self.S_inv[action_id] = A_inv - numerator / denominator
            else:
                # 数値不安定な場合は直接計算
                self.S_inv[action_id] = np.linalg.inv(self.S[action_id])

            # 平均ベクトルを更新: m_a = S_a^{-1} * (S_a * m_a + beta * x * r)
            self.m[action_id] = np.dot(
                self.S_inv[action_id], np.dot(self.S[action_id], self.m[action_id]) + self.beta * context * reward
            )

    @property
    def algorithm_name(self) -> str:
        """アルゴリズム名を返す."""
        return f"ThompsonSamplingRanking(alpha={self.alpha}, beta={self.beta})"

    def get_action_confidence_intervals(self, context: np.ndarray, confidence: float = 0.95) -> dict:
        """各actionの報酬期待値の信頼区間を取得する（デバッグ用）.

        Parameters
        ----------
        context : np.ndarray
            コンテキスト特徴量
        confidence : float
            信頼区間の信頼度

        Returns
        -------
        dict
            action_id -> (lower_bound, upper_bound, mean) の辞書
        """
        from scipy.stats import norm

        z_score = norm.ppf((1 + confidence) / 2)
        intervals = {}

        for action_id in range(self.num_actions):
            mean = np.dot(context, self.m[action_id])
            var = np.dot(context, np.dot(self.S_inv[action_id], context)) / self.beta
            std = np.sqrt(var)

            intervals[action_id] = (
                mean - z_score * std,  # lower_bound
                mean + z_score * std,  # upper_bound
                mean,  # mean
            )

        return intervals
