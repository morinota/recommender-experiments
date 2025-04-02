import itertools
from pathlib import Path
import random
from typing import Callable, Literal, Optional, TypedDict
from joblib import Parallel, delayed
import numpy as np
from obp.dataset import SyntheticBanditDataset, logistic_reward_function
from obp.ope import ReplayMethod, InverseProbabilityWeighting, BaseOffPolicyEstimator
import polars as pl
from obp.policy import IPWLearner, NNPolicyLearner, Random, LogisticTS, BernoulliTS
import pydantic
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from recommender_experiments.service.opl.two_tower_nn_model import PolicyByTwoTowerModel
from recommender_experiments.service.simulator.opl_simulator import run_opl_multiple_simulations_in_parallel
from recommender_experiments.service.utils.expected_reward_functions import ContextFreeBinary, ContextAwareBinary


class BanditFeedbackDict(TypedDict):
    n_rounds: int  # ラウンド数
    n_actions: int  # アクション数s
    context: np.ndarray  # 文脈 (shape: (n_rounds, dim_context))
    action_context: np.ndarray  # アクション特徴量 (shape: (n_actions, dim_action_features))
    action: np.ndarray  # 実際に選択されたアクション (shape: (n_rounds,))
    position: Optional[np.ndarray]  # ポジション (shape: (n_rounds,) or None)
    reward: np.ndarray  # 報酬 (shape: (n_rounds,))
    expected_reward: np.ndarray  # 期待報酬 (shape: (n_rounds, n_actions))
    pi_b: np.ndarray  # データ収集方策 P(a|x) (shape: (n_rounds, n_actions))
    pscore: np.ndarray  # 傾向スコア (shape: (n_rounds,))


def _dot_product_based_logging_policy(
    context: np.ndarray, action_context: np.ndarray, random_state: int = None
) -> np.ndarray:
    """ユーザとニュースのコンテキストを考慮し、
    コンテキストベクトル $x$ とアイテムコンテキストベクトル $e$ の内積が最も大きいニュースを
    確率0.7で推薦し、その他のニュースを均等に確率0.1で推薦する確率的方策。
    返り値:
    """
    n_rounds = context.shape[0]
    n_actions = action_context.shape[0]
    epsilon = 0.1

    # 内積を計算
    scores = context @ action_context.T  # shape: (n_rounds, n_actions)

    # 各ラウンドで最もスコアが高いアクションのindexを取得
    selected_actions = np.argmax(scores, axis=1)  # shape: (n_rounds,)

    # 確率的方策: 確率0.1で全てのアクションを一様ランダムに選択し、確率0.6で最もスコアが高いアクションを決定的に選択
    action_dist = np.full((n_rounds, n_actions), epsilon / n_actions)
    action_dist[np.arange(n_rounds), selected_actions] = 1.0 - epsilon + epsilon / n_actions
    return action_dist


def simulate_opl_by_reward_scale(result_path: Path) -> None:
    """期待報酬 E_{p(r|a,x)}[r] のスケールの違いが、オフライン学習に与える影響を実験する"""
    # 実験パラメータ
    n_simulations = 30
    n_actions_list = [50]
    dim_context_list = [50]
    n_rounds_train_list = [2000, 5000, 10000, 20000, 50000]
    n_rounds_test_list = [10000]
    batch_size_list = [2000]
    n_epochs_list = [10]
    expected_reward_scale_list = [
        (0.0, 0.001),
        (0.0, 0.005),
        (0.0, 0.01),
        (0.0, 0.05),
        (0.0, 0.1),
        (0.0, 0.3),
        (0.0, 0.5),
        (0.0, 0.7),
        (0.0, 0.9),
    ]
    expected_reward_settings = ["my_context_aware"]
    new_policy_settings = ["obp_nn"]
    logging_policy_functions = [_dot_product_based_logging_policy]
    n_jobs = 5

    # シミュレーションの実行
    results = run_opl_multiple_simulations_in_parallel(
        n_simulations,
        n_actions_list,
        dim_context_list,
        n_rounds_train_list,
        n_rounds_test_list,
        batch_size_list,
        n_epochs_list,
        expected_reward_scale_list,
        expected_reward_settings,
        new_policy_settings,
        n_jobs,
        logging_policy_functions,
    )

    # シミュレーション結果の保存
    result_df = pl.DataFrame([result.model_dump() for result in results])
    # result_df.write_csv(result_path)


def main() -> None:
    results_dir = Path("logs/cost_benefit_experiment/")
    results_dir.mkdir(parents=True, exist_ok=True)

    simulate_opl_by_reward_scale(results_dir / "opl_result_by_reward_scale.csv")


if __name__ == "__main__":
    main()
