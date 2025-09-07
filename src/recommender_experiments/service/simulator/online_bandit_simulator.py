"""オンラインバンディット学習シミュレータ."""

from typing import Dict

from sklearn.utils import check_random_state

from ..algorithms.bandit_algorithm_interface import BanditAlgorithmInterface, OnlineEvaluationResults
from ..environment.bandit_environment import BanditEnvironmentInterface


class OnlineBanditSimulator:
    """オンラインバンディット学習シミュレータ.

    BanditEnvironmentInterfaceを環境として使用し、
    バンディットアルゴリズムのオンライン学習性能を評価する。

    Parameters
    ----------
    environment : BanditEnvironmentInterface
        シミュレーション環境
    random_state : int
        乱数シード
    """

    def __init__(self, environment: BanditEnvironmentInterface, random_state: int = 42):
        self.environment = environment
        self.random_state = random_state
        self.random_ = check_random_state(random_state)

    def evaluate_online_learning(
        self,
        algorithm: BanditAlgorithmInterface,
        num_trials: int,
    ) -> OnlineEvaluationResults:
        """単一アルゴリズムのオンライン学習性能を評価する.

        Parameters
        ----------
        algorithm : BanditAlgorithmInterface
            評価対象のバンディットアルゴリズム
        num_trials : int
            実行する試行数

        Returns
        -------
        OnlineEvaluationResults
            評価結果
        """
        # アルゴリズムをリセット
        algorithm.reset()

        # 結果格納用オブジェクト
        results = OnlineEvaluationResults(algorithm.algorithm_name)

        # 環境をリセット
        self.environment.reset()

        for trial in range(num_trials):
            # 1. 環境からコンテキストと利用可能actionを取得
            context, available_actions = self.environment.get_context_and_available_actions(trial)

            # 2. アルゴリズムが行動を選択
            # available_actionsから最大k個選択（kは環境設定による）
            k = len(available_actions)  # 最大選択可能数
            selected_actions = algorithm.select_actions(context=context, available_actions=available_actions, k=k)

            # 3. 環境から報酬を観測
            rewards = self.environment.get_rewards(context, selected_actions, trial)

            # 4. アルゴリズムを更新
            algorithm.update(context, selected_actions, rewards)

            # 5. 評価指標を計算
            instant_reward = sum(rewards)
            optimal_reward = self.environment.get_optimal_reward(context, available_actions, k)
            instant_regret = optimal_reward - instant_reward

            # 6. 結果を記録
            results.add_trial_result(
                selected_actions=selected_actions, instant_regret=instant_regret, instant_reward=instant_reward
            )

        return results

    def compare_algorithms(
        self,
        algorithms: Dict[str, BanditAlgorithmInterface],
        num_trials: int,
    ) -> Dict[str, OnlineEvaluationResults]:
        """複数のアルゴリズムの性能を比較する.

        Parameters
        ----------
        algorithms : Dict[str, BanditAlgorithmInterface]
            アルゴリズム名とアルゴリズムオブジェクトの辞書
        num_trials : int
            実行する試行数

        Returns
        -------
        Dict[str, OnlineEvaluationResults]
            アルゴリズム名をキーとした評価結果の辞書
        """
        comparison_results = {}

        for name, algorithm in algorithms.items():
            print(f"Evaluating {name}...")
            results = self.evaluate_online_learning(algorithm=algorithm, num_trials=num_trials)
            comparison_results[name] = results

        return comparison_results
