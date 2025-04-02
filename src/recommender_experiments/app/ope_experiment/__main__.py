from loguru import logger
import typer
from pathlib import Path
from joblib import delayed, Parallel
import numpy as np
from pandas import DataFrame
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import yaml
from obp.dataset import logistic_reward_function, SyntheticBanditDataset, OpenBanditDataset
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
    ReplayMethod,
)
from obp.policy import IPWLearner

app = typer.Typer(pretty_exceptions_enable=False)


def load_hyperparameters(config_path: str = "./configs/ope_experiment/hyperparams.yml") -> dict:
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
        ReplayMethod(),
        DirectMethod(),
        InverseProbabilityWeighting(),
        # SelfNormalizedInverseProbabilityWeighting(),
        DoublyRobust(),
        # SelfNormalizedDoublyRobust(),
        # SwitchDoublyRobustTuning(lambdas=[10, 50, 100, 500, 1000, 5000, 10000, np.inf]),
        # DoublyRobustWithShrinkageTuning(
        #     lambdas=[10, 50, 100, 500, 1000, 5000, 10000, np.inf]
        # ),
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
    1回のシミュレーションを実行し、OPE推定量で方策の性能を推定する。

    Args:
        i (int): シミュレーションのインデックス。
        n_actions (int): アクションの数。
        dim_context (int): 文脈ベクトルの次元数。
        beta (float): データの逆温度パラメータ。
        n_rounds (int): サンプルサイズ。
        base_model_for_evaluation_policy (str): 評価方策のMLモデル。
        base_model_for_reg_model (str): 回帰モデルのMLモデル。
        hyperparams (dict): モデルのハイパーパラメータ。
        base_model_dict (dict): 使用可能なベースモデルの辞書。
        ope_estimators (list): 評価するOPE推定量のリスト。
        random_state (int): ランダムシード。
    Returns:
        dict: OPE推定量ごとの相対誤差を格納した辞書。
    """
    # データ収集方策によって集められるはずの、擬似バンディットデータを定義
    dataset = SyntheticBanditDataset(
        n_actions=n_actions,
        dim_context=dim_context,
        reward_function=logistic_reward_function,
        beta=beta,
        random_state=i,
    )
    # 学習用とテスト用のバンディットデータを生成
    bandit_feedback_train = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
    bandit_feedback_test = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
    logger.debug(f"bandit_feedback['action']: {bandit_feedback_train['action'].shape}")
    logger.debug(f"bandit_feedback['expected_reward']: {bandit_feedback_train['expected_reward'].shape}")
    logger.debug(f"dataset.n_actions: {dataset.n_actions}")

    # 評価方策のモデル構造を定義
    evaluation_policy = IPWLearner(
        n_actions=dataset.n_actions,
        base_classifier=base_model_dict[base_model_for_evaluation_policy](
            **hyperparams[base_model_for_evaluation_policy]
        ),
    )
    logger.debug(f"evaluation_policy: {evaluation_policy}")

    # データ収集方策によって集められた学習データを用いて、評価方策をオフライン学習(たぶん逆傾向スコア重み付け!)
    evaluation_policy.fit(
        context=bandit_feedback_train["context"],
        action=bandit_feedback_train["action"],
        reward=bandit_feedback_train["reward"],
        pscore=bandit_feedback_train["pscore"],
    )

    # テストデータの各タイムステップに対して、評価方策による行動選択の確率分布を推論
    ## predictメソッドは決定的な行動選択。sample_actionメソッドは確率的な行動選択。
    action_dist = evaluation_policy.predict_proba(context=bandit_feedback_test["context"])
    ground_truth_policy_value = dataset.calc_ground_truth_policy_value(
        # あ、擬似データだから、各(context, action)ペアに対する期待報酬 E[r|a,x] が既知なんだ...!!
        expected_reward=bandit_feedback_test["expected_reward"],
        # 上に加えて、評価方策の行動選択確率 P(a|x) を渡せば、累積報酬の期待値が算出できる
        action_dist=action_dist,
    )
    print(f"ground_truth_policy_value: {ground_truth_policy_value}")

    # これはOPE推定量の準備(DM推定量のための報酬予測モデル)
    regression_model = RegressionModel(
        n_actions=dataset.n_actions,
        action_context=dataset.action_context,
        base_model=base_model_dict[base_model_for_reg_model](**hyperparams[base_model_for_reg_model]),
    )
    estimated_rewards_by_reg_model = regression_model.fit_predict(
        context=bandit_feedback_test["context"],
        action=bandit_feedback_test["action"],
        reward=bandit_feedback_test["reward"],
        n_folds=3,
        random_state=random_state,
    )

    # オフライン評価(各OPE推定量で評価方策のオンライン性能を予測 & ground-truthと比較?)
    ope = OffPolicyEvaluation(bandit_feedback=bandit_feedback_test, ope_estimators=ope_estimators)

    metric_i = ope.evaluate_performance_of_estimators(
        ground_truth_policy_value=ground_truth_policy_value,
        # 評価方策の行動選択確率 P(a|x)は、IPS推定量などの値を計算するために渡す
        # (データ収集方策の行動選択確率は、コンストラクタで渡し済み）
        action_dist=action_dist,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        metric="relative-ee",  # OPE推定量の性能を評価するための指標
    )
    estimated_policy_value_by_ope = ope.estimate_policy_values(
        action_dist=action_dist, estimated_rewards_by_reg_model=estimated_rewards_by_reg_model
    )
    print(f"estimated_policy_value_by_ope: {estimated_policy_value_by_ope}")

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
        beta (float): データ収集方策の逆温度パラメータ。
        n_rounds (int): サンプルサイズ。
        base_model_for_evaluation_policy (str): 評価方策のMLモデル。
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
            simulation_idx,
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
        for simulation_idx in np.arange(n_runs)
    )


def compile_results(processed: list, ope_estimators: list, log_path: Path = Path("./logs")) -> None:
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
    n_rounds: int = typer.Option(10000, help="収集されたバンディットデータのサンプル数。"),
    n_actions: int = typer.Option(10, help="アクションの数。"),
    dim_context: int = typer.Option(5, help="文脈ベクトルの次元。"),
    beta: float = typer.Option(
        3.0, help="データ収集方策の逆温度パラメータ。大きいほど決定的な方策に、小さいほど探索的な方策になる。"
    ),
    base_model_for_evaluation_policy: str = typer.Option(
        ..., help="評価方策に使用するMLモデル。", prompt=True, metavar="logistic_regression|lightgbm|random_forest"
    ),
    base_model_for_reg_model: str = typer.Option(
        ..., help="回帰モデルに使用するMLモデル。", prompt=True, metavar="logistic_regression|lightgbm|random_forest"
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
