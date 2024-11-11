import typer
from pathlib import Path
from joblib import delayed, Parallel
import numpy as np
from pandas import DataFrame
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import yaml

from obp.dataset import logistic_reward_function, SyntheticBanditDataset
from obp.ope import (
    DirectMethod,
    InverseProbabilityWeighting,
    SelfNormalizedInverseProbabilityWeighting,
    DoublyRobust,
    SelfNormalizedDoublyRobust,
    SwitchDoublyRobustTuning,
    DoublyRobustWithShrinkageTuning,
    OffPolicyEvaluation,
    RegressionModel,
)
from obp.policy import IPWLearner

app = typer.Typer()

# hyperparameters of the regression model used in model dependent OPE estimators
with open("./configs/ope_experiment/hyperparams.yml", "rb") as f:
    hyperparams = yaml.safe_load(f)

base_model_dict = {
    "logistic_regression": LogisticRegression,
    "lightgbm": GradientBoostingClassifier,
    "random_forest": RandomForestClassifier,
}

ope_estimators = [
    DirectMethod(),
    InverseProbabilityWeighting(),
    SelfNormalizedInverseProbabilityWeighting(),
    DoublyRobust(),
    SelfNormalizedDoublyRobust(),
    SwitchDoublyRobustTuning(lambdas=[10, 50, 100, 500, 1000, 5000, 10000, np.inf]),
    DoublyRobustWithShrinkageTuning(
        lambdas=[10, 50, 100, 500, 1000, 5000, 10000, np.inf]
    ),
]


@app.command()
def main(
    n_runs: int = typer.Option(1, help="Number of simulations in the experiment."),
    n_rounds: int = typer.Option(10000, help="Sample size of logged bandit data."),
    n_actions: int = typer.Option(10, help="Number of actions."),
    dim_context: int = typer.Option(5, help="Dimensions of context vectors."),
    beta: float = typer.Option(
        3.0, help="Inverse temperature parameter for behavior policy."
    ),
    base_model_for_evaluation_policy: str = typer.Option(
        ...,
        help="ML model for evaluation policy.",
        prompt=True,
        case_sensitive=False,
        metavar="logistic_regression|lightgbm|random_forest",
    ),
    base_model_for_reg_model: str = typer.Option(
        ...,
        help="ML model for regression model.",
        prompt=True,
        case_sensitive=False,
        metavar="logistic_regression|lightgbm|random_forest",
    ),
    n_jobs: int = typer.Option(1, help="Maximum number of concurrent jobs."),
    random_state: int = typer.Option(12345, help="Random seed."),
):
    print(
        f"Running with config: n_runs={n_runs}, n_rounds={n_rounds}, n_actions={n_actions}, ..."
    )

    def process(i: int):
        dataset = SyntheticBanditDataset(
            n_actions=n_actions,
            dim_context=dim_context,
            reward_function=logistic_reward_function,
            beta=beta,
            random_state=i,
        )
        evaluation_policy = IPWLearner(
            n_actions=dataset.n_actions,
            base_classifier=base_model_dict[base_model_for_evaluation_policy](
                **hyperparams[base_model_for_evaluation_policy]
            ),
        )
        bandit_feedback_train = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
        bandit_feedback_test = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
        evaluation_policy.fit(
            context=bandit_feedback_train["context"],
            action=bandit_feedback_train["action"],
            reward=bandit_feedback_train["reward"],
            pscore=bandit_feedback_train["pscore"],
        )
        action_dist = evaluation_policy.predict_proba(
            context=bandit_feedback_test["context"]
        )
        regression_model = RegressionModel(
            n_actions=dataset.n_actions,
            action_context=dataset.action_context,
            base_model=base_model_dict[base_model_for_reg_model](
                **hyperparams[base_model_for_reg_model]
            ),
        )
        estimated_rewards_by_reg_model = regression_model.fit_predict(
            context=bandit_feedback_test["context"],
            action=bandit_feedback_test["action"],
            reward=bandit_feedback_test["reward"],
            n_folds=3,
            random_state=random_state,
        )
        ope = OffPolicyEvaluation(
            bandit_feedback=bandit_feedback_test, ope_estimators=ope_estimators
        )
        metric_i = ope.evaluate_performance_of_estimators(
            ground_truth_policy_value=dataset.calc_ground_truth_policy_value(
                expected_reward=bandit_feedback_test["expected_reward"],
                action_dist=action_dist,
            ),
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            metric="relative-ee",
        )
        return metric_i

    processed = Parallel(n_jobs=n_jobs, verbose=50)(
        [delayed(process)(i) for i in np.arange(n_runs)]
    )
    metric_dict = {est.estimator_name: {} for est in ope_estimators}
    for i, metric_i in enumerate(processed):
        for estimator_name, relative_ee_ in metric_i.items():
            metric_dict[estimator_name][i] = relative_ee_
    results_df = DataFrame(metric_dict).describe().T.round(6)

    print("=" * 45)
    print(f"random_state={random_state}")
    print("-" * 45)
    print(results_df[["mean", "std"]])
    print("=" * 45)

    log_path = Path("./logs")
    log_path.mkdir(exist_ok=True, parents=True)
    results_df.to_csv(log_path / "evaluation_of_ope_results.csv")


if __name__ == "__main__":
    app()
