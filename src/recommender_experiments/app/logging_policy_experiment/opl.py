# off-policy learning用の実験コード


from typing import Optional, TypedDict
import numpy as np
import polars as pl
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from obp.dataset import logistic_reward_function, SyntheticBanditDataset
from obp.policy import IPWLearner, NNPolicyLearner, Random, LogisticTS, BernoulliTS

from recommender_experiments.app.logging_policy_experiment.expected_reward_function import expected_reward_function


class BanditFeedbackDict(TypedDict):
    n_rounds: int  # ラウンド数
    n_actions: int  # アクション数
    context: np.ndarray  # 文脈 (shape: (n_rounds, dim_context))
    action_context: np.ndarray  # アクション特徴量 (shape: (n_actions, dim_action_features))
    action: np.ndarray  # 実際に選択されたアクション (shape: (n_rounds,))
    position: Optional[np.ndarray]  # ポジション (shape: (n_rounds,) or None)
    reward: np.ndarray  # 報酬 (shape: (n_rounds,))
    expected_reward: np.ndarray  # 期待報酬 (shape: (n_rounds, n_actions))
    pi_b: np.ndarray  # データ収集方策 P(a|x) (shape: (n_rounds, n_actions))
    pscore: np.ndarray  # 傾向スコア (shape: (n_rounds,))


def setup_dataset(n_actions: int, dim_context: int, beta: float, random_state: int):
    """Set up the synthetic bandit dataset."""
    return SyntheticBanditDataset(
        n_actions=n_actions,
        dim_context=dim_context,
        # reward_function=logistic_reward_function,
        reward_function=expected_reward_function,
        beta=beta,
        random_state=random_state,
    )


def train_policies(
    bandit_feedback_train: BanditFeedbackDict,
    dataset: SyntheticBanditDataset,
    base_model: str,
    off_policy_objective: str,
    dim_context: int,
    random_state: int,
    hyperparams: dict,
) -> tuple[IPWLearner, NNPolicyLearner, LogisticTS, BernoulliTS]:
    """学習用データセットを使って、評価方策の最適化を試みる"""
    base_model_dict = {
        "logistic_regression": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
    }

    ipw_learner = IPWLearner(
        n_actions=dataset.n_actions, base_classifier=base_model_dict[base_model](**hyperparams["base_model"])
    )
    ipw_learner.fit(
        context=bandit_feedback_train["context"],
        action=bandit_feedback_train["action"],
        reward=bandit_feedback_train["reward"],
        pscore=bandit_feedback_train["pscore"],
    )

    nn_policy_learner = NNPolicyLearner(
        n_actions=dataset.n_actions,
        dim_context=dim_context,
        off_policy_objective=off_policy_objective,
        hidden_layer_size=(100,),
        activation="relu",
        solver="adam",
        random_state=random_state,
    )
    # nn_policy_learner.fit(
    #     context=bandit_feedback_train["context"],
    #     action=bandit_feedback_train["action"],
    #     reward=bandit_feedback_train["reward"],
    #     pscore=bandit_feedback_train["pscore"],
    # )

    # contexual bandtのLogisticTSを定義してパラメータ更新させてみる
    logistic_ts = LogisticTS(dim=dim_context, n_actions=dataset.n_actions, random_state=random_state)
    logistic_ts.update_params(action=0, reward=1.0, context=np.array([[0.0, 0.0, 0.0, 0.0, 0.0]]))

    # contextfree banditのBernoulliTSを定義してパラメータ更新させてみる
    contextfree_ts = BernoulliTS(n_actions=dataset.n_actions, random_state=random_state)
    contextfree_ts.update_params(action=0, reward=1.0)

    print(contextfree_ts)

    return ipw_learner, nn_policy_learner, logistic_ts, contextfree_ts


def evaluate_policies(
    dataset: SyntheticBanditDataset,
    bandit_feedback_test: BanditFeedbackDict,
    random_policy: Random,
    ipw_learner: IPWLearner,
    nn_policy_learner: NNPolicyLearner,
) -> pl.DataFrame:
    """学習用データセットで学習した方策を、テスト用データセットで評価する"""
    # 各評価方策をテスト用データセットに適用した時のアクション選択確率を算出
    random_action_dist = random_policy.compute_batch_action_dist(n_rounds=len(bandit_feedback_test["context"]))
    ipw_action_dist = ipw_learner.predict(context=bandit_feedback_test["context"])
    nn_action_dist = nn_policy_learner.predict_proba(context=bandit_feedback_test["context"])

    # 真の即時報酬の期待値を使って、各方策の真の累積報酬の期待値を算出
    random_policy_value = dataset.calc_ground_truth_policy_value(
        expected_reward=bandit_feedback_test["expected_reward"], action_dist=random_action_dist
    )
    ipw_policy_value = dataset.calc_ground_truth_policy_value(
        expected_reward=bandit_feedback_test["expected_reward"], action_dist=ipw_action_dist
    )
    nn_policy_value = dataset.calc_ground_truth_policy_value(
        expected_reward=bandit_feedback_test["expected_reward"], action_dist=nn_action_dist
    )

    return pl.DataFrame(
        {
            "policy": ["random_policy", "ipw_learner", "nn_policy_learner"],
            "true_policy_value": [random_policy_value, ipw_policy_value, nn_policy_value],
        }
    )


def main() -> None:
    n_rounds = 10000
    n_actions = 10
    dim_context = 5
    beta = -3
    base_model = "logistic_regression"
    off_policy_objective = "ipw"
    random_state = 12345

    # データセットをセットアップ
    dataset = setup_dataset(n_actions=n_actions, dim_context=dim_context, beta=beta, random_state=random_state)

    # 学習用とテスト用のデータセットを生成
    bandit_feedback_train = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
    bandit_feedback_test = dataset.obtain_batch_bandit_feedback(n_rounds=10000)

    # ランダム方策を定義
    random_policy = Random(n_actions=dataset.n_actions, random_state=random_state)

    # 学習用データセットを使って、評価方策を最適化
    ipw_learner, nn_policy_learner, logistic_ts, contextfree_ts = train_policies(
        bandit_feedback_train=bandit_feedback_train,
        dataset=dataset,
        base_model=base_model,
        off_policy_objective=off_policy_objective,
        dim_context=dim_context,
        random_state=random_state,
        hyperparams={"base_model": {"max_iter": 1000}},
    )

    # テスト用データセットの全ラウンドに、各方策を適用した時の真の累積報酬の期待値を算出
    result_df = evaluate_policies(
        dataset=dataset,
        bandit_feedback_test=bandit_feedback_test,
        random_policy=random_policy,
        ipw_learner=ipw_learner,
        nn_policy_learner=nn_policy_learner,
    )
    print(result_df)


if __name__ == "__main__":
    main()
