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


def load_hyperparameters(
    config_path: str = "./configs/ope_experiment/hyperparams.yml",
) -> dict:
    """
    設定ファイルからハイパーパラメータを読み込む。

    Args:
        config_path (str): ハイパーパラメータ設定ファイルのパス。デフォルトは "./configs/ope_experiment/hyperparams.yml"

    Returns:
        dict: ハイパーパラメータを格納した辞書。
    """
    with open(config_path, "rb") as f:
        return yaml.safe_load(f)


def get_base_model_dict() -> dict:
    """
    使用可能なベースモデルを格納した辞書を取得する。

    Returns:
        dict: モデル名をキーとし、モデルクラスを値とする辞書。
    """
    return {
        "logistic_regression": LogisticRegression,
        "lightgbm": GradientBoostingClassifier,
        "random_forest": RandomForestClassifier,
    }


def get_ope_estimators() -> list:
    """
    評価するOPE推定量のリストを取得する。

    Returns:
        list: OPE推定量のインスタンスのリスト。
    """
    return [
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


def run_single_simulation(
    i: int,
    n_actions: int,
    dim_context: int,
    beta: float,
    n_rounds: int,
    base_model_for_evaluation_policy: str,
    base_model_for_reg_model: str,
    hyperparams: dict,
    base_model_dict: dict,
    ope_estimators: list,
    random_state: int,
) -> dict:
    """
    1回のシミュレーションを実行し、OPE推定量でポリシーの価値を評価する。

    Args:
        i (int): シミュレーションのインデックス。
        n_actions (int): アクションの数。
        dim_context (int): 文脈ベクトルの次元数。
        beta (float): 行動ポリシーの逆温度パラメータ。
        n_rounds (int): サンプルサイズ。
        base_model_for_evaluation_policy (str): 評価ポリシーのMLモデル。
        base_model_for_reg_model (str): 回帰モデルのMLモデル。
        hyperparams (dict): モデルのハイパーパラメータ。
        base_model_dict (dict): 使用可能なベースモデルの辞書。
        ope_estimators (list): 評価するOPE推定量のリスト。
        random_state (int): ランダムシード。
    Returns:
        dict: 推定量ごとの相対誤差を格納した辞書。
    """
    dataset = SyntheticBanditDataset(
        n_actions=n_actions,
        dim_context=dim_context,
        reward_function=logistic_reward_function,
        beta=beta,
        random_state=i,
    )
    bandit_feedback_train = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
    bandit_feedback_test = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)

    evaluation_policy = IPWLearner(
        n_actions=dataset.n_actions,
        base_classifier=base_model_dict[base_model_for_evaluation_policy](
            **hyperparams[base_model_for_evaluation_policy]
        ),
    )
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


def run_simulations_in_parallel(
    n_runs: int,
    n_jobs: int,
    n_actions: int,
    dim_context: int,
    beta: float,
    n_rounds: int,
    base_model_for_evaluation_policy: str,
    base_model_for_reg_model: str,
    hyperparams: dict,
    base_model_dict: dict,
    ope_estimators: list,
    random_state: int,
) -> list:
    """
    複数のシミュレーションを並列で実行する。

    Args:
        n_runs (int): シミュレーションの総数。
        n_jobs (int): 同時に実行する最大ジョブ数。
        n_actions (int): アクションの数。
        dim_context (int): 文脈ベクトルの次元数。
        beta (float): 行動ポリシーの逆温度パラメータ。
        n_rounds (int): サンプルサイズ。
        base_model_for_evaluation_policy (str): 評価ポリシーのMLモデル。
        base_model_for_reg_model (str): 回帰モデルのMLモデル。
        hyperparams (dict): モデルのハイパーパラメータ。
        base_model_dict (dict): 使用可能なベースモデルの辞書。
        ope_estimators (list): 評価するOPE推定量のリスト。
        random_state (int): ランダムシード。

    Returns:
        list: 各シミュレーションの結果を格納したリスト。
    """
    return Parallel(n_jobs=n_jobs, verbose=50)(
        delayed(run_single_simulation)(
            i,
            n_actions,
            dim_context,
            beta,
            n_rounds,
            base_model_for_evaluation_policy,
            base_model_for_reg_model,
            hyperparams,
            base_model_dict,
            ope_estimators,
            random_state,
        )
        for i in np.arange(n_runs)
    )


def compile_results(
    processed: list, ope_estimators: list, log_path: Path = Path("./logs")
) -> None:
    """
    シミュレーション結果を集計し、CSV形式で保存する。

    Args:
        processed (list): 各シミュレーションの結果リスト。
        ope_estimators (list): OPE推定量のリスト。
        log_path (Path): 結果保存ディレクトリのパス。デフォルトは "./logs"。
    """
    metric_dict = {est.estimator_name: {} for est in ope_estimators}
    for i, metric_i in enumerate(processed):
        for estimator_name, relative_ee_ in metric_i.items():
            metric_dict[estimator_name][i] = relative_ee_
    results_df = DataFrame(metric_dict).describe().T.round(6)

    print("=" * 45)
    print(results_df[["mean", "std"]])
    print("=" * 45)

    log_path.mkdir(exist_ok=True, parents=True)
    results_df.to_csv(log_path / "evaluation_of_ope_results.csv")


@app.command()
def main(
    n_runs: int = typer.Option(1, help="実験のシミュレーション回数。"),
    n_rounds: int = typer.Option(
        10000, help="ログされたバンディットデータのサンプル数。"
    ),
    n_actions: int = typer.Option(10, help="アクションの数。"),
    dim_context: int = typer.Option(5, help="文脈ベクトルの次元。"),
    beta: float = typer.Option(3.0, help="行動ポリシーの逆温度パラメータ。"),
    base_model_for_evaluation_policy: str = typer.Option(
        ...,
        help="評価ポリシーに使用するMLモデル。",
        prompt=True,
        metavar="logistic_regression|lightgbm|random_forest",
    ),
    base_model_for_reg_model: str = typer.Option(
        ...,
        help="回帰モデルに使用するMLモデル。",
        prompt=True,
        metavar="logistic_regression|lightgbm|random_forest",
    ),
    n_jobs: int = typer.Option(1, help="並列で実行する最大ジョブ数。"),
    random_state: int = typer.Option(12345, help="ランダムシード。"),
) -> None:
    """
    メイン関数。指定されたパラメータでシミュレーションを実行し、結果を保存する。
    """
    hyperparams = load_hyperparameters()
    base_model_dict = get_base_model_dict()
    ope_estimators = get_ope_estimators()

    processed = run_simulations_in_parallel(
        n_runs,
        n_jobs,
        n_actions,
        dim_context,
        beta,
        n_rounds,
        base_model_for_evaluation_policy,
        base_model_for_reg_model,
        hyperparams,
        base_model_dict,
        ope_estimators,
        random_state,
    )

    compile_results(processed, ope_estimators)


if __name__ == "__main__":
    app()
