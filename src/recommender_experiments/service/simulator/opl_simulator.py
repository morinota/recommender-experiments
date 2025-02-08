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
from recommender_experiments.service.opl.two_tower_nn_model import (
    TwoTowerNNPolicyLearner,
)
from recommender_experiments.service.utils.expected_reward_functions import (
    ContextFreeBinary,
    ContextAwareBinary,
)
from recommender_experiments.service.utils.logging_policies import random_policy


class OPLSimulationResult(pydantic.BaseModel):
    simulation_idx: int
    n_actions: int
    dim_context: int
    n_rounds_train: int
    n_rounds_test: int
    batch_size: int
    expected_reward_lower: float
    expected_reward_upper: float
    expected_reward_setting: Literal["my_context_free", "my_context_aware", "linear"]
    new_policy_setting: Literal["two_tower_nn", "obp_nn"]
    new_policy_value: float


def run_opl_single_simulation(
    n_simulations: int,
    n_actions: int,
    dim_context: int,
    action_context: np.ndarray,
    n_rounds_train: int,
    n_rounds_test: int,
    batch_size: int,
    logging_policy_function: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
    n_epochs: int = 200,
    expected_reward_lower: float = 0.0,
    expected_reward_upper: float = 0.5,
    expected_reward_setting: Literal[
        "my_context_free", "my_context_aware", "linear"
    ] = "my_context_aware",
    new_policy_setting: Literal["two_tower_nn", "obp_nn"] = "two_tower_nn",
    learning_rate_init: float = 0.00001,
) -> list[OPLSimulationResult]:
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

    results = []
    for simulation_idx in range(n_simulations):
        # データ収集方策によって集められるはずの、擬似バンディットデータの設定を定義
        dataset = SyntheticBanditDataset(
            n_actions=n_actions,
            dim_context=dim_context,
            reward_type="binary",
            reward_function=reward_function,
            behavior_policy_function=logging_policy_function,
            random_state=simulation_idx,
            action_context=action_context,
        )

        # 新方策のためのNNモデルを初期化
        if new_policy_setting == "obp_nn":
            new_policy = NNPolicyLearner(
                n_actions=n_actions,
                dim_context=dim_context,
                off_policy_objective="ipw",
                learning_rate_init=learning_rate_init,
                batch_size=batch_size,
                max_iter=n_epochs,
                random_state=simulation_idx,
            )
        elif new_policy_setting == "two_tower_nn":
            new_policy = TwoTowerNNPolicyLearner(
                dim_context=dim_context,
                dim_action_features=action_context.shape[1],
                dim_two_tower_embedding=100,
                off_policy_objective="ipw",
                learning_rate_init=learning_rate_init,
                batch_size=batch_size,
                max_iter=n_epochs,
                random_state=simulation_idx,
            )
        # 学習前の新方策の真の性能を確認
        bandit_feedback_test = dataset.obtain_batch_bandit_feedback(n_rounds_test)
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

        # データ収集方策で集めたデータ(学習用)で、新方策のためのNNモデルのパラメータを更新
        bandit_feedback_train = dataset.obtain_batch_bandit_feedback(n_rounds_train)
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

        # 学習後の新方策の真の性能を確認
        if new_policy_setting == "obp_nn":
            test_action_dist = new_policy.predict_proba(
                context=bandit_feedback_test["context"],
            )
        elif new_policy_setting == "two_tower_nn":
            test_action_dist = new_policy.predict_proba(
                context=bandit_feedback_test["context"],
                action_context=bandit_feedback_test["action_context"],
            )
        new_policy_value = dataset.calc_ground_truth_policy_value(
            expected_reward=bandit_feedback_test["expected_reward"],
            action_dist=test_action_dist,
        )

        # 結果を保存
        results.append(
            OPLSimulationResult(
                **{
                    "simulation_idx": simulation_idx,
                    "n_actions": n_actions,
                    "dim_context": dim_context,
                    "n_rounds_train": n_rounds_train,
                    "n_rounds_test": n_rounds_test,
                    "batch_size": batch_size,
                    "expected_reward_lower": expected_reward_lower,
                    "expected_reward_upper": expected_reward_upper,
                    "expected_reward_setting": expected_reward_setting,
                    "new_policy_setting": new_policy_setting,
                    "new_policy_value": new_policy_value,
                }
            )
        )

    return results


def run_opl_multiple_simulations_in_parallel(
    n_simulations: int,
    n_actions_list: list[int] = [5, 10],
    dim_context_list: list[int] = [5, 10],
    n_rounds_train_list: list[int] = [50, 100],
    n_rounds_test_list: list[int] = [50, 100],
    batch_size_list: list[int] = [32, 64],
    n_epochs_list: list[int] = [5, 10],
    expected_reward_scale_list: list[tuple[float, float]] = [
        (0.1, 0.2),
        (0.1, 0.5),
    ],
    expected_reward_settings: list[str] = ["my_context_aware"],
    new_policy_settings: list[str] = ["two_tower_nn"],
    n_jobs: int = 5,
    logging_policy_functions: list[
        Callable[[np.ndarray, np.ndarray, int], np.ndarray]
    ] = None,
) -> list[OPLSimulationResult]:
    if logging_policy_functions is None:
        logging_policy_functions = [random_policy]

    # シミュレーション設定の組み合わせを作成
    simulate_configs = list(
        itertools.product(
            n_actions_list,
            dim_context_list,
            n_rounds_train_list,
            n_rounds_test_list,
            batch_size_list,
            n_epochs_list,
            expected_reward_scale_list,
            expected_reward_settings,
            new_policy_settings,
            logging_policy_functions,
        )
    )
    print(f"simulate_configs: {simulate_configs}")

    parallel_results: list[list[OPLSimulationResult]] = Parallel(n_jobs=n_jobs)(
        delayed(run_opl_single_simulation)(
            n_simulations=n_simulations,
            n_actions=n_actions,
            dim_context=dim_context,
            # 期待報酬関数の設定の都合で、action_contextの次元数をdim_contextと同じにしてる。
            action_context=np.random.random((n_actions, dim_context)),
            n_rounds_train=n_rounds_train,
            n_rounds_test=n_rounds_test,
            batch_size=batch_size,
            n_epochs=n_epochs,
            logging_policy_function=logging_policy_function,
            expected_reward_setting=expected_reward_setting,
            expected_reward_lower=expected_reward_lower,
            expected_reward_upper=expected_reward_upper,
            new_policy_setting=new_policy_setting,
            learning_rate_init=0.00001,
        )
        for _, (
            n_actions,
            dim_context,
            n_rounds_train,
            n_rounds_test,
            batch_size,
            n_epochs,
            (expected_reward_lower, expected_reward_upper),
            expected_reward_setting,
            new_policy_setting,
            logging_policy_function,
        ) in enumerate(simulate_configs)
    )

    # list[list[OPLSimulationResult]] -> list[OPLSimulationResult]
    flattened_results = list(itertools.chain.from_iterable(parallel_results))
    return flattened_results
