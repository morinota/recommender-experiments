import random
from typing import Callable, Literal, Optional, TypedDict
import numpy as np
from obp.dataset import SyntheticBanditDataset
from obp.ope import ReplayMethod, InverseProbabilityWeighting, BaseOffPolicyEstimator
import polars as pl
from obp.policy import IPWLearner, NNPolicyLearner, Random, LogisticTS, BernoulliTS
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from recommender_experiments.service.opl.two_tower_nn_model import (
    TwoTowerNNPolicyLearner,
)
from recommender_experiments.service.utils.expected_reward_functions import (
    context_free_binary,
)


class BanditFeedbackDict(TypedDict):
    n_rounds: int  # ラウンド数
    n_actions: int  # アクション数s
    context: np.ndarray  # 文脈 (shape: (n_rounds, dim_context))
    action_context: (
        np.ndarray
    )  # アクション特徴量 (shape: (n_actions, dim_action_features))
    action: np.ndarray  # 実際に選択されたアクション (shape: (n_rounds,))
    position: Optional[np.ndarray]  # ポジション (shape: (n_rounds,) or None)
    reward: np.ndarray  # 報酬 (shape: (n_rounds,))
    expected_reward: np.ndarray  # 期待報酬 (shape: (n_rounds, n_actions))
    pi_b: np.ndarray  # データ収集方策 P(a|x) (shape: (n_rounds, n_actions))
    pscore: np.ndarray  # 傾向スコア (shape: (n_rounds,))


def _logging_policy(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: int = None,
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
    action_dist[np.arange(n_rounds), selected_actions] = (
        1.0 - epsilon + epsilon / n_actions
    )
    return action_dist


def _run_single_simulation(
    simulate_idx: int,
    n_rounds_train: int,
    n_rounds_test: int,
    n_actions: int,
    dim_context: int,
    action_context: np.ndarray,
    reward_function: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
    logging_policy_function: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
) -> dict:
    # データ収集方策によって集められるはずの、擬似バンディットデータの設定を定義
    dataset = SyntheticBanditDataset(
        n_actions=n_actions,
        dim_context=dim_context,
        reward_type="binary",
        reward_function=reward_function,
        behavior_policy_function=logging_policy_function,
        random_state=simulate_idx,
        action_context=action_context,
    )

    # 収集されるバンディットフィードバックデータを生成
    bandit_feedback_train = dataset.obtain_batch_bandit_feedback(n_rounds_train)
    bandit_feedback_test = dataset.obtain_batch_bandit_feedback(n_rounds_test)

    # 新方策のためのNNモデルを初期化
    new_policy = TwoTowerNNPolicyLearner(
        dim_context=dim_context,
        dim_action_features=bandit_feedback_train["action_context"].shape[1],
        dim_two_tower_embedding=10,
    )

    # データ収集方策で集めたデータ(学習用)で、新方策のためのNNモデルのパラメータを更新
    new_policy.fit(
        context=bandit_feedback_train["context"],
        action_context=bandit_feedback_train["action_context"],
        action=bandit_feedback_train["action"],
        reward=bandit_feedback_train["reward"],
        pscore=bandit_feedback_train["pscore"],
    )

    # データ収集方策で集めたデータ(評価用)で、新方策の性能を確認
    test_action_dist = new_policy.predict_proba(
        context=bandit_feedback_test["context"],
        action_context=bandit_feedback_test["action_context"],
    )
    ground_truth_new_policy_value = dataset.calc_ground_truth_policy_value(
        expected_reward=bandit_feedback_test["expected_reward"],
        action_dist=test_action_dist,
    )

    return {
        "simulation_idx": simulate_idx,
        "n_rounds_train": n_rounds_train,
        "n_rounds_test": n_rounds_test,
        "ground_truth_new_policy_value": ground_truth_new_policy_value,
    }


def main() -> None:
    print(
        _run_single_simulation(
            simulate_idx=0,
            n_rounds_train=10000,
            n_rounds_test=10000,
            n_actions=4,
            dim_context=50,
            action_context=np.random.random((4, 50)),
            reward_function=context_free_binary,
            logging_policy_function=_logging_policy,
        )
    )


if __name__ == "__main__":
    main()
