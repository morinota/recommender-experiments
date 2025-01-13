import itertools
from pathlib import Path
import random
from typing import Callable, Literal, Optional, TypedDict
import numpy as np
from obp.dataset import SyntheticBanditDataset, logistic_reward_function
from obp.ope import ReplayMethod, InverseProbabilityWeighting, BaseOffPolicyEstimator
import polars as pl
from obp.policy import IPWLearner, NNPolicyLearner, Random, LogisticTS, BernoulliTS
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from recommender_experiments.service.opl.two_tower_nn_model import (
    TwoTowerNNPolicyLearner,
)
from recommender_experiments.service.utils.expected_reward_functions import (
    ContextFreeBinary,
    ContextAwareBinary,
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
    logging_policy_function: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
    expected_reward_lower: float,
    expected_reward_upper: float,
    # is_expected_reward_context_aware: bool = False,
    expected_reward_setting: Literal[
        "my_context_free", "my_context_aware", "linear"
    ] = "my_context_aware",
    new_policy_setting: Literal["two_tower_nn", "obp_nn"] = "two_tower_nn",
) -> dict:
    # 期待報酬関数を設定
    if expected_reward_setting == "my_context_aware":
        reward_function_generator = ContextAwareBinary(
            expected_reward_lower, expected_reward_upper
        )
        reward_function = reward_function_generator.get_function()
    elif expected_reward_setting == "my_context_free":
        reward_function_generator = ContextFreeBinary(
            expected_reward_lower, expected_reward_upper
        )
        reward_function = reward_function_generator.get_function()
    elif expected_reward_setting == "linear":
        reward_function = logistic_reward_function

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
    bandit_feedback_test = dataset.obtain_batch_bandit_feedback(n_rounds_test)

    # 新方策のためのNNモデルを初期化
    if new_policy_setting == "obp_nn":
        new_policy = NNPolicyLearner(
            n_actions=n_actions,
            dim_context=dim_context,
            random_state=simulate_idx,
            off_policy_objective="ipw",
        )
    elif new_policy_setting == "two_tower_nn":
        new_policy = TwoTowerNNPolicyLearner(
            dim_context=dim_context,
            dim_action_features=action_context.shape[1],
            dim_two_tower_embedding=100,
            off_policy_objective="ipw",
        )
    # 学習前の真の性能を確認
    if new_policy_setting == "obp_nn":
        test_action_dist = new_policy.predict_proba(
            context=bandit_feedback_test["context"],
        )
    elif new_policy_setting == "two_tower_nn":
        test_action_dist = new_policy.predict_proba(
            context=bandit_feedback_test["context"],
            action_context=bandit_feedback_test["action_context"],
        )
    policy_value_before_fit = dataset.calc_ground_truth_policy_value(
        expected_reward=bandit_feedback_test["expected_reward"],
        action_dist=test_action_dist,
    )
    print(f"policy_value_before_fit: {policy_value_before_fit}")

    # 新方策の性能の推移を保存するlist
    new_policy_value_list: list[tuple] = []

    # 学習データ数を10分割して、段階的にfittingを行い、新方策の性能の推移を記録する
    splitted_n_rounds_train = n_rounds_train // 10
    for i in range(10):
        bandit_feedback_train = dataset.obtain_batch_bandit_feedback(
            splitted_n_rounds_train
        )
        # データ収集方策で集めたデータ(学習用)で、新方策のためのNNモデルのパラメータを更新
        if new_policy_setting == "obp_nn":
            new_policy.fit(
                context=bandit_feedback_train["context"],
                action=bandit_feedback_train["action"],
                reward=bandit_feedback_train["reward"],
                pscore=bandit_feedback_train["pscore"],
            )
        elif new_policy_setting == "two_tower_nn":
            new_policy.fit(
                context=bandit_feedback_train["context"],
                action_context=bandit_feedback_train["action_context"],
                action=bandit_feedback_train["action"],
                reward=bandit_feedback_train["reward"],
                pscore=bandit_feedback_train["pscore"],
            )

        # データ収集方策で集めたデータ(評価用)で、学習後の新方策の真の性能を確認
        if new_policy_setting == "obp_nn":
            test_action_dist = new_policy.predict_proba(
                context=bandit_feedback_test["context"],
            )
        elif new_policy_setting == "two_tower_nn":
            test_action_dist = new_policy.predict_proba(
                context=bandit_feedback_test["context"],
                action_context=bandit_feedback_test["action_context"],
            )
        ground_truth_new_policy_value = dataset.calc_ground_truth_policy_value(
            expected_reward=bandit_feedback_test["expected_reward"],
            action_dist=test_action_dist,
        )
        print(
            f"ground_truth_new_policy_value(at {(i+1) * splitted_n_rounds_train} data): {ground_truth_new_policy_value}"
        )
        new_policy_value_list.append(
            ((i + 1) * splitted_n_rounds_train, ground_truth_new_policy_value)
        )

    return {
        "simulation_idx": simulate_idx,
        "n_actions": n_actions,
        "n_rounds_train": n_rounds_train,
        "n_rounds_test": n_rounds_test,
        "expected_reward_lower": expected_reward_lower,
        "expected_reward_upper": expected_reward_upper,
        "ground_truth_new_policy_value": ground_truth_new_policy_value,
        "new_policy_value_list": new_policy_value_list,
    }


from loguru import logger
from pandera.polars import DataFrameSchema
from pandera.typing.polars import DataFrame


class SimulationResult(DataFrameSchema):
    n_actions: int
    n_rounds_train: int
    n_rounds_test: int
    expected_reward_lower: float
    expected_reward_upper: float
    expected_reward_setting: Literal["my_context_free", "my_context_aware", "linear"]
    new_policy_setting: Literal["two_tower_nn", "obp_nn"]
    ground_truth_new_policy_value: float


def _run_multiple_simulations(
    n_actions_list: list[int] = [5, 10],
    dim_context_list: list[int] = [5, 10],
    expected_reward_scale_list: list[tuple[float, float]] = [
        (0.1, 0.2),
        (0.1, 0.5),
    ],
) -> DataFrame[SimulationResult]:
    # 固定値を定義
    fixed_n_round_train = 1000
    fixed_n_round_test = 1000

    # シミュレーション設定の組み合わせを作成
    simulate_configs = list(
        itertools.product(
            n_actions_list,
            dim_context_list,
            expected_reward_scale_list,
        )
    )
    print(f"simulate_configs: {simulate_configs}")

    results = []
    for config in simulate_configs:
        n_actions, dim_context, (expected_reward_lower, expected_reward_upper) = config

        # 期待報酬関数の設定の都合で、action_contextの次元数をdim_contextと同じにしているが、別に必須ではない。
        action_context = np.random.random((n_actions, dim_context))
        result_dict = _run_single_simulation(
            simulate_idx=0,
            n_rounds_train=fixed_n_round_train,
            n_rounds_test=fixed_n_round_test,
            n_actions=n_actions,
            dim_context=dim_context,
            action_context=action_context,
            logging_policy_function=_logging_policy,
            expected_reward_lower=expected_reward_lower,
            expected_reward_upper=expected_reward_upper,
            expected_reward_setting="my_context_aware",
            new_policy_setting="two_tower_nn",
        )

        # result_dictのnew_policy_value_listを分解して、dfの1行にする
        for n_round_train, new_policy_value in result_dict["new_policy_value_list"]:
            results.append(
                {
                    "n_actions": n_actions,
                    "n_rounds_train": fixed_n_round_train,
                    "n_rounds_test": fixed_n_round_test,
                    "expected_reward_lower": expected_reward_lower,
                    "expected_reward_upper": expected_reward_upper,
                    "expected_reward_setting": "my_context_aware",
                    "new_policy_setting": "two_tower_nn",
                    "n_round_train": n_round_train,
                    "new_policy_value": new_policy_value,
                }
            )

    return pl.DataFrame(results)


def main() -> None:
    # 実験パラメータ
    n_actions_list = [5, 10, 20, 50, 100, 1000]
    dim_context_list = [5, 10, 20, 50, 100]
    expected_reward_scale_list = [
        (0.1, 0.9),
        (0.1, 0.7),
        (0.1, 0.5),
        (0.1, 0.3),
        (0.1, 0.2),
    ]
    result_df = _run_multiple_simulations()

    # 結果の保存
    results_dir = Path("logs/cost_benefit_experiment/")
    results_dir.mkdir(parents=True, exist_ok=True)
    result_df.write_csv(results_dir / "result_df.csv")

    # print(
    #     _run_single_simulation(
    #         simulate_idx=0,
    #         n_rounds_train=10000,
    #         n_rounds_test=1000,
    #         n_actions=4,
    #         dim_context=50,
    #         action_context=np.random.random((4, 50)),
    #         logging_policy_function=_logging_policy,
    #         expected_reward_lower=0.05,
    #         expected_reward_upper=0.10,
    #         expected_reward_setting="my_context_aware",
    #         # new_policy_setting="obp_nn",
    #         new_policy_setting="two_tower_nn",
    #     )
    # )


if __name__ == "__main__":
    main()
