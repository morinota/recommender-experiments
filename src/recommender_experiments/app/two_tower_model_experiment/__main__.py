import itertools
from pathlib import Path
from joblib import delayed, Parallel
import random
from typing import Callable, Literal, Optional, TypedDict
import numpy as np
from obp.dataset import SyntheticBanditDataset, logistic_reward_function
from obp.ope import ReplayMethod, InverseProbabilityWeighting, BaseOffPolicyEstimator
import polars as pl
from obp.policy import IPWLearner, NNPolicyLearner, Random, LogisticTS, BernoulliTS
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import torch
from tqdm import tqdm
from recommender_experiments.service.opl.shared_parameter_nn_model import (
    SharedParameterNNPolicyLearner,
)
from recommender_experiments.service.opl.two_tower_nn_model import (
    TwoTowerNNPolicyLearner,
)
from recommender_experiments.service.utils import logging_policies
from recommender_experiments.service.utils.expected_reward_functions import (
    ContextFreeBinary,
    ContextAwareBinary,
)
from loguru import logger
from pandera.polars import DataFrameSchema
from pandera.typing.polars import DataFrame


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


def _run_single_simulation(
    n_rounds_train: int,
    n_rounds_test: int,
    n_actions: int,
    dim_context: int,
    action_context: np.ndarray,
    logging_policy_function: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
    expected_reward_lower: float,
    expected_reward_upper: float,
    expected_reward_setting: Literal[
        "my_context_free", "my_context_aware", "linear"
    ] = "my_context_aware",
    learning_rate_init: float = 0.0001,
    should_ips_estimate: bool = True,
    new_policy_setting: Literal[
        "two_tower_nn", "obp_nn", "shared_parameter_nn"
    ] = "two_tower_nn",
) -> list[dict]:
    # 期待報酬関数を設定
    if expected_reward_setting == "my_context_aware":
        reward_function_generator = ContextAwareBinary(
            expected_reward_lower, expected_reward_upper, False
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
        action_context=action_context,
    )

    # 収集されるバンディットフィードバックデータを生成
    bandit_feedback_test = dataset.obtain_batch_bandit_feedback(n_rounds_test)

    # データ収集方策の性能を確認
    logging_policy_value = dataset.calc_ground_truth_policy_value(
        expected_reward=bandit_feedback_test["expected_reward"],
        action_dist=bandit_feedback_test["pi_b"],
    )
    logger.debug(f"{logging_policy_value=}")

    # 新方策のためのNNモデルを初期化
    if new_policy_setting == "two_tower_nn":
        new_policy = TwoTowerNNPolicyLearner(
            dim_context=dim_context,
            dim_action_features=action_context.shape[1],
            dim_two_tower_embedding=100,
            off_policy_objective="ipw",
            learning_rate_init=learning_rate_init,
            is_embedding_normed=False,
            softmax_temprature=1,
        )
    elif new_policy_setting == "shared_parameter_nn":
        new_policy = SharedParameterNNPolicyLearner(
            dim_context=dim_context + action_context.shape[1],
            off_policy_objective="ipw",
            learning_rate_init=learning_rate_init,
        )
    elif new_policy_setting == "obp_nn":
        new_policy = NNPolicyLearner(
            n_actions=n_actions,
            dim_context=dim_context,
            off_policy_objective="ipw",
            learning_rate_init=learning_rate_init,
        )
    else:
        raise ValueError(f"new_policy_setting: {new_policy_setting} is not supported")

    # # 学習前の真の性能を確認
    # test_action_dist = new_policy.predict_proba(
    #     context=bandit_feedback_test["context"],
    #     action_context=bandit_feedback_test["action_context"],
    # )
    # policy_value_before_fit = dataset.calc_ground_truth_policy_value(
    #     expected_reward=bandit_feedback_test["expected_reward"],
    #     action_dist=test_action_dist,
    # )
    # logger.debug(f"{policy_value_before_fit=}")

    # 学習データ数を10分割して、段階的にfittingを行い、新方策の性能の推移を記録する
    new_policy_value_by_n_train = {}
    splitted_n_rounds_train = 2000
    for num_of_train_data in range(2000, n_rounds_train, 4000):
        bandit_feedback_train = dataset.obtain_batch_bandit_feedback(
            splitted_n_rounds_train
        )
        # データ収集方策で集めたデータ(学習用)で、two-towerモデルのパラメータを更新
        if new_policy_setting == "two_tower_nn":
            new_policy.fit(
                context=bandit_feedback_train["context"],
                action_context=bandit_feedback_train["action_context"],
                action=bandit_feedback_train["action"],
                reward=bandit_feedback_train["reward"],
                pscore=bandit_feedback_train["pscore"] if should_ips_estimate else None,
            )

            test_action_dist = new_policy.predict_proba(
                context=bandit_feedback_test["context"],
                action_context=bandit_feedback_test["action_context"],
            )

        elif new_policy_setting == "shared_parameter_nn":
            new_policy.fit(
                context=bandit_feedback_train["context"],
                action_context=bandit_feedback_train["action_context"],
                action=bandit_feedback_train["action"],
                reward=bandit_feedback_train["reward"],
                pscore=bandit_feedback_train["pscore"] if should_ips_estimate else None,
            )

            test_action_dist = new_policy.predict_proba(
                context=bandit_feedback_test["context"],
                action_context=bandit_feedback_test["action_context"],
            )

        elif new_policy_setting == "obp_nn":
            new_policy.fit(
                context=bandit_feedback_train["context"],
                action=bandit_feedback_train["action"],
                reward=bandit_feedback_train["reward"],
                pscore=bandit_feedback_train["pscore"] if should_ips_estimate else None,
            )

            test_action_dist = new_policy.predict_proba(
                context=bandit_feedback_test["context"]
            )

        else:
            raise ValueError(
                f"new_policy_setting: {new_policy_setting} is not supported"
            )

        ground_truth_new_policy_value = dataset.calc_ground_truth_policy_value(
            expected_reward=bandit_feedback_test["expected_reward"],
            action_dist=test_action_dist,
        )
        # 学習データ数をキーとして、新方策の性能を記録
        new_policy_value_by_n_train[num_of_train_data] = ground_truth_new_policy_value
        logger.debug(
            f"n_rounds_train: {num_of_train_data}, new_policy_value: {ground_truth_new_policy_value}, relative value: {ground_truth_new_policy_value/logging_policy_value}"
        )

    # 分割した学習データ数ごとの新方策の性能をlistで返す
    return [
        {
            "n_actions": n_actions,
            "n_rounds_train": _n_rounds_train,
            "n_rounds_test": n_rounds_test,
            "expected_reward_lower": expected_reward_lower,
            "expected_reward_upper": expected_reward_upper,
            "should_ips_estimate": should_ips_estimate,
            "expected_reward_setting": expected_reward_setting,
            "new_policy_value": _policy_value,
        }
        for _n_rounds_train, _policy_value in new_policy_value_by_n_train.items()
    ]


class SimulationResult(DataFrameSchema):
    n_actions: int
    n_rounds_train: int
    n_rounds_test: int
    expected_reward_lower: float
    expected_reward_upper: float
    expected_reward_setting: Literal["my_context_free", "my_context_aware", "linear"]
    new_policy_setting: Literal["two_tower_nn", "obp_nn"]
    new_policy_value: float
    relative_policy_value: float  # new_policy_value / expected_reward_upper


def _run_simulations_in_parallel(
    n_actions_list: list[int],
    dim_context_list: list[int],
    expected_reward_scale_list: list[tuple[float, float]] = [(0.1, 0.3)],
    expected_reward_settings: list[str] = ["my_context_aware"],
    n_rounds_train_list: int = list[5000, 10000, 15000, 20000, 25000],
    fixed_n_round_test: int = 1000,
    n_jobs: int = 3,
) -> DataFrame[SimulationResult]:
    """
    複数のシミュレーションを並列で実行する
    """

    should_ips_estimate_list = [True, False]
    # シミュレーション設定の組み合わせを作成
    simulate_configs = list(
        itertools.product(
            n_actions_list,
            dim_context_list,
            expected_reward_scale_list,
            expected_reward_settings,
            should_ips_estimate_list,
            n_rounds_train_list,
        )
    )
    logger.debug(f"simulate_configs: {simulate_configs}")

    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_single_simulation)(
            n_rounds_train=n_rounds_train,
            n_rounds_test=fixed_n_round_test,
            n_actions=n_actions,
            dim_context=dim_context,
            # 期待報酬関数の設定の都合で、action_contextの次元数をdim_contextと同じにしてる
            action_context=np.random.random((n_actions, dim_context)),
            logging_policy_function=logging_policies.context_aware_stochastic_policy,
            expected_reward_lower=expected_reward_lower,
            expected_reward_upper=expected_reward_upper,
            expected_reward_setting=expected_reward_setting,
            learning_rate_init=0.00001,
            should_ips_estimate=should_ips_estimate,
        )
        for (
            n_actions,
            dim_context,
            (expected_reward_lower, expected_reward_upper),
            expected_reward_setting,
            should_ips_estimate,
            n_rounds_train,
        ) in tqdm(simulate_configs, desc="simulation progress")
    )

    return pl.DataFrame(results)


def main() -> None:
    # 実験パラメータ
    n_actions_list = [5, 10, 20, 40]
    dim_context_list = [50]
    expected_reward_scale_list = [(0.0, 0.4)]
    # expected_reward_settings = ["my_context_aware", "my_context_free", "linear"]
    expected_reward_settings = ["my_context_aware"]

    # シミュレーションの実行
    result_df = _run_simulations_in_parallel(
        n_actions_list=n_actions_list,
        dim_context_list=dim_context_list,
        expected_reward_scale_list=expected_reward_scale_list,
        expected_reward_settings=expected_reward_settings,
        n_rounds_train_list=[25000],
        fixed_n_round_test=1000,
    )

    # シミュレーション結果の保存
    results_dir = Path("logs/two_tower_model_experiment")
    results_dir.mkdir(parents=True, exist_ok=True)
    result_df.write_csv(results_dir / "result_df.csv")


if __name__ == "__main__":
    # main()
    n_actions = 10
    torch.autograd.set_detect_anomaly(True)
    _run_single_simulation(
        n_rounds_train=40000,
        n_rounds_test=1000,
        n_actions=n_actions,
        dim_context=50,
        action_context=np.random.random((n_actions, 50)),
        logging_policy_function=logging_policies.random_policy,
        expected_reward_lower=0.01,
        expected_reward_upper=0.4,
        expected_reward_setting="my_context_aware",
        should_ips_estimate=True,
        new_policy_setting="two_tower_nn",
        # new_policy_setting="shared_parameter_nn",
        # new_policy_setting="obp_nn",
    )
