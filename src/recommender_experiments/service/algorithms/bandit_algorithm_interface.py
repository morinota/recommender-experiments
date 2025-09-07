"""バンディットアルゴリズムの共通インターフェース."""

from abc import ABC, abstractmethod

import numpy as np


class BanditAlgorithmInterface(ABC):
    """バンディットアルゴリズムの共通インターフェース.

    このインターフェースを実装することで、様々なバンディットアルゴリズム
    （Thompson Sampling, UCB, ε-greedy等）を統一的に扱えます。
    """

    @abstractmethod
    def select_actions(self, context: np.ndarray, available_actions: np.ndarray, k: int) -> list[int]:
        """利用可能なactionの中からk個の行動を選択する.

        Parameters
        ----------
        context : np.ndarray
            コンテキスト特徴量 (dim_context,)
        available_actions : np.ndarray
            利用可能なaction IDの配列 (num_available_actions,)
        k : int
            選択する行動数（ランキング長）

        Returns
        -------
        list[int]
            選択された行動のリスト（長さk）
        """
        pass

    @abstractmethod
    def update(self, context: np.ndarray, selected_actions: list[int], rewards: list[float]) -> None:
        """観測された報酬をもとにアルゴリズムのパラメータを更新する.

        Parameters
        ----------
        context : np.ndarray
            コンテキスト特徴量 (dim_context,)
        selected_actions : list[int]
            選択された行動のリスト
        rewards : list[float]
            各行動に対応する報酬のリスト
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """アルゴリズムの状態をリセットする."""
        pass

    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        """アルゴリズム名を返す."""
        pass


class OnlineEvaluationResults:
    """オンライン学習評価の結果を格納するデータクラス.

    Attributes
    ----------
    algorithm_name : str
        評価したアルゴリズム名
    num_trials : int
        実行した試行数
    cumulative_regret : list[float]
        累積regret（各試行時点での累積値）
    instant_regret : list[float]
        瞬時regret（各試行での単発regret）
    cumulative_reward : list[float]
        累積報酬（各試行時点での累積値）
    instant_reward : list[float]
        瞬時報酬（各試行での単発報酬）
    selected_actions_history : list[list[int]]
        選択行動の履歴（trial_idx -> [action1, action2, ...]）
    """

    def __init__(self, algorithm_name: str):
        self.algorithm_name = algorithm_name
        self.num_trials = 0
        self.cumulative_regret: list[float] = []
        self.instant_regret: list[float] = []
        self.cumulative_reward: list[float] = []
        self.instant_reward: list[float] = []
        self.selected_actions_history: list[list[int]] = []

    def add_trial_result(self, selected_actions: list[int], instant_regret: float, instant_reward: float):
        """1試行分の結果を追加する.

        Parameters
        ----------
        selected_actions : list[int]
            選択された行動のリスト
        instant_regret : float
            その試行での瞬時regret
        instant_reward : float
            その試行での瞬時報酬
        """
        self.num_trials += 1
        self.selected_actions_history.append(selected_actions.copy())
        self.instant_regret.append(instant_regret)
        self.instant_reward.append(instant_reward)

        # 累積値を計算
        prev_cumulative_regret = self.cumulative_regret[-1] if self.cumulative_regret else 0.0
        prev_cumulative_reward = self.cumulative_reward[-1] if self.cumulative_reward else 0.0

        self.cumulative_regret.append(prev_cumulative_regret + instant_regret)
        self.cumulative_reward.append(prev_cumulative_reward + instant_reward)

    def get_final_cumulative_regret(self) -> float:
        """最終的な累積regretを取得する."""
        return self.cumulative_regret[-1] if self.cumulative_regret else 0.0

    def get_final_cumulative_reward(self) -> float:
        """最終的な累積報酬を取得する."""
        return self.cumulative_reward[-1] if self.cumulative_reward else 0.0

    def get_average_regret(self) -> float:
        """平均regretを取得する."""
        return self.get_final_cumulative_regret() / self.num_trials if self.num_trials > 0 else 0.0
