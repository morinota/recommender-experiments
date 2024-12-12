import numpy as np
from obp.dataset import SyntheticBanditDataset
from obp.ope import ReplayMethod, InverseProbabilityWeighting

from recommender_experiments.app.logging_policy_experiment.expected_reward_function import (
    expected_reward_function,
)
from recommender_experiments.app.logging_policy_experiment.logging_policy import (
    logging_policy_function_stochastic,
)


# 評価方策(target policy)として、常にニュース0を選択する方策を設定
def target_policy(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: int = None,
    recommend_arm_idx: int = 0,
) -> np.ndarray:
    """入力値として、各ラウンドの文脈とアクションの文脈を受け取り、各アクションの選択確率を返す関数
    - 返り値のshape: (n_rounds, n_actions, len_list)
    - 今回は、推薦アイテムリストの長さは1(len_list=1)なので、(n_rounds, n_actions, 1)の配列を返す
    """
    n_rounds = context.shape[0]
    n_actions = action_context.shape[0]

    # recommend_arm_idx番目のアクションの選択確率を1.0、それ以外のアクションの選択確率を0.0とする
    p_scores = np.zeros(n_actions)
    p_scores[recommend_arm_idx] = 1.0

    # 返り値の形式に整形: (n_rounds, n_actions)の配列で、各行が各ラウンドでのアクションの選択確率を表す
    action_dist = np.zeros((n_rounds, n_actions, 1))
    action_dist[:, :, 0] = p_scores

    assert np.allclose(
        action_dist.sum(axis=1), 1.0
    ), "各ラウンドでの全てのアクションの選択確率の合計は1.0である必要があります"

    return action_dist


# ニュース推薦用の文脈付きバンディットデータセットを生成
dataset = SyntheticBanditDataset(
    n_actions=4,  # 推薦候補ニュースの数
    dim_context=5,
    reward_type="binary",
    reward_function=expected_reward_function,
    # behavior_policy_function=logging_policy_function_deterministic,
    behavior_policy_function=logging_policy_function_stochastic,
    random_state=12345,
)

# バンディットデータを生成
bandit_feedback: dict = dataset.obtain_batch_bandit_feedback(n_rounds=1000)

# バンディットデータの中身を確認
print(f"{bandit_feedback.keys()=}")
print(f"{bandit_feedback['n_rounds']=}")
print(f"{bandit_feedback['n_actions']=}")
# print(f"{bandit_feedback['context']=}")
# print(f"{bandit_feedback['action_context']=}")
# print(f"{bandit_feedback['action']=}")
# print(f"{bandit_feedback['position']=}")
# print(f"{bandit_feedback['reward']=}")
# print(f"{bandit_feedback['expected_reward']=}")
# print(f"{bandit_feedback['pi_b']=}")
# print(f"{bandit_feedback['pscore']=}")

# 評価方策を使って、ログデータ(bandit_feedback)に対する行動選択確率を計算
target_policy_action_dist = target_policy(
    context=bandit_feedback["context"],
    action_context=bandit_feedback["action_context"],
    recommend_arm_idx=0,
)

# 事前に設定した真の期待報酬E[r|x,a]を使って、評価方策の真の性能を計算
ground_truth_policy_value = dataset.calc_ground_truth_policy_value(
    expected_reward=bandit_feedback["expected_reward"],
    action_dist=target_policy_action_dist,
)
print(f"{ground_truth_policy_value=}")

# OPE推定量を準備(naive推定量とIPS推定量)
naive_estimator = ReplayMethod()
ips_estimator = InverseProbabilityWeighting()

# OPE推定量を使って、評価方策の性能を推定
## naive推定量
estimated_policy_value_by_naive = naive_estimator.estimate_policy_value(
    reward=bandit_feedback["reward"],
    action=bandit_feedback["action"],
    action_dist=target_policy_action_dist,
)
print(
    f"naive推定量(Replay Method)による評価方策の性能の推定値: {estimated_policy_value_by_naive}"
)

## IPS推定量
estimated_policy_value_by_ips = ips_estimator.estimate_policy_value(
    action=bandit_feedback["action"],  # 各ラウンドで観測された行動
    reward=bandit_feedback["reward"],  # 各ラウンドで観測された報酬
    pscore=bandit_feedback[
        "pscore"
    ],  # 各ラウンドで、データ収集方策がそのアクションを選択する確率
    action_dist=target_policy_action_dist,  # 各ラウンドでの評価方策のアクション選択確率
)
print(f"IPS推定量による評価方策の性能の推定値: {estimated_policy_value_by_ips}")
