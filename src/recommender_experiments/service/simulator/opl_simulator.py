import itertools
from pathlib import Path
import random
from tqdm_joblib import tqdm_joblib

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
    PolicyByTwoTowerModel,
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
    off_policy_learning_method: Literal["ips", "dr", "regression_based"]
    logging_policy_value: float
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
    off_policy_learning_method: str = "dr",
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
                off_policy_objective=off_policy_learning_method,
                learning_rate_init=learning_rate_init,
                batch_size=batch_size,
                max_iter=n_epochs,
                random_state=simulation_idx,
            )
        elif new_policy_setting == "two_tower_nn":
            new_policy = PolicyByTwoTowerModel(
                dim_context_features=dim_context,
                dim_action_features=action_context.shape[1],
                dim_two_tower_embedding=100,
                off_policy_objective=off_policy_learning_method,
                learning_rate_init=learning_rate_init,
                batch_size=batch_size,
                max_iter=n_epochs,
                random_state=simulation_idx,
            )
        # 性能評価用のデータを取得
        bandit_feedback_test = dataset.obtain_batch_bandit_feedback(n_rounds_test)
        # データ収集方策の性能を確認
        logging_policy_value = dataset.calc_ground_truth_policy_value(
            expected_reward=bandit_feedback_test["expected_reward"],
            action_dist=bandit_feedback_test["pi_b"],
        )
        # 新方策の学習前の性能を確認
        if isinstance(new_policy, NNPolicyLearner):
            test_action_dist = new_policy.predict_proba(
                context=bandit_feedback_test["context"],
            )
        elif isinstance(new_policy, PolicyByTwoTowerModel):
            test_action_dist = new_policy.predict_proba(
                context=bandit_feedback_test["context"],
                action_context=bandit_feedback_test["action_context"],
            )
        policy_value_before_fit = dataset.calc_ground_truth_policy_value(
            expected_reward=bandit_feedback_test["expected_reward"],
            action_dist=test_action_dist,
        )

        # データ収集方策で集めたデータ(学習用)で、新方策のためのNNモデルのパラメータを更新
        bandit_feedback_train = dataset.obtain_batch_bandit_feedback(n_rounds_train)
        if isinstance(new_policy, NNPolicyLearner):
            new_policy.fit(
                context=bandit_feedback_train["context"],
                action=bandit_feedback_train["action"],
                reward=bandit_feedback_train["reward"],
                pscore=bandit_feedback_train["pscore"],
            )
        elif isinstance(new_policy, PolicyByTwoTowerModel):
            new_policy.fit(
                bandit_feedback_train=bandit_feedback_train,
                bandit_feedback_test=bandit_feedback_test,
            )

        # 学習後の新方策の真の性能を確認
        if isinstance(new_policy, NNPolicyLearner):
            test_action_dist = new_policy.predict_proba(
                context=bandit_feedback_test["context"],
            )
        elif isinstance(new_policy, PolicyByTwoTowerModel):
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
                    "logging_policy_value": logging_policy_value,
                    "off_policy_learning_method": off_policy_learning_method,
                }
            )
        )
        print(f"simulation_idx: {simulation_idx}, new_policy_value: {new_policy_value}")
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
    off_policy_learning_methods: list[str] = ["dr"],
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
            off_policy_learning_methods,
        )
    )
    # シミュレーションを並列で実行
    with tqdm_joblib(tqdm(desc="Simulating OPL", total=len(simulate_configs))):
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
                off_policy_learning_method=off_policy_learning_method,
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
                off_policy_learning_method,
            ) in enumerate(simulate_configs)
        )

    # list[list[OPLSimulationResult]] -> list[OPLSimulationResult]にflatten
    flattened_results = list(itertools.chain.from_iterable(parallel_results))
    return flattened_results


if __name__ == "__main__":
    # Arrange
    n_simulations = 1
    n_actions = 10
    dim_context = 50
    n_rounds_train = 20000
    n_rounds_test = 2000
    batch_size = 200
    n_epochs = 200
    action_context = np.random.random(size=(n_actions, dim_context))
    # データ収集方策は簡単のため、一様ランダムな方策を指定
    logging_policy_function = lambda context, action_context, random_state: np.full(
        (context.shape[0], n_actions), 1.0 / n_actions
    )
    # 真の期待報酬 E_{p(r|x,a)}[r] の設定
    expected_reward_lower = 0.0
    expected_reward_upper = 0.5
    expected_reward_setting = "my_context_aware"
    # 新方策の設定
    new_policy_setting = "two_tower_nn"
    off_policy_learning_method = "regression_based"

    # Act
    actual = run_opl_single_simulation(
        n_simulations=n_simulations,
        n_actions=n_actions,
        dim_context=dim_context,
        action_context=action_context,
        n_rounds_train=n_rounds_train,
        n_rounds_test=n_rounds_test,
        batch_size=batch_size,
        n_epochs=n_epochs,
        logging_policy_function=logging_policy_function,
        expected_reward_setting=expected_reward_setting,
        expected_reward_lower=expected_reward_lower,
        expected_reward_upper=expected_reward_upper,
        new_policy_setting=new_policy_setting,
        off_policy_learning_method=off_policy_learning_method,
    )

    print(actual)

    # Act
    actual = run_opl_multiple_simulations_in_parallel(
        n_simulations=1,
        n_actions_list=[10],
        dim_context_list=[50],
        n_rounds_train_list=[2000, 5000, 10000, 15000, 20000],
        n_rounds_test_list=[2000],
        batch_size_list=[200],
        n_epochs_list=[200],
        expected_reward_scale_list=[(0.0, 0.5)],
        expected_reward_settings=["linear"],
        new_policy_settings=["two_tower_nn"],
        logging_policy_functions=[logging_policy_function],
        off_policy_learning_methods=["dr", "ips", "regression_based"],
        n_jobs=5,
    )
    result_df = pl.DataFrame([result.model_dump() for result in actual])
    print(result_df)
