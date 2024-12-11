import numpy as np
from obp.dataset import SyntheticBanditDataset
from obp.ope import (
    ReplayMethod,
    InverseProbabilityWeighting,
    SelfNormalizedInverseProbabilityWeighting,
    DoublyRobust,
    OffPolicyEvaluation,
    RegressionModel,
)
from sklearn.ensemble import RandomForestClassifier


def expected_reward_function(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: int = None,
) -> np.ndarray:
    """(アクションa, 文脈x)の各組み合わせに対する期待報酬 E[r|x,a] を定義する関数
    今回の場合は、推薦候補4つの記事を送った場合の報酬rの期待値を、文脈xに依存しない固定値として設定する
    ニュース0: 0.2, ニュース1: 0.15, ニュース2: 0.1, ニュース3: 0.05
    """
    n_rounds = context.shape[0]
    n_actions = action_context.shape[0]

    # 固定の期待報酬を設定 (n_actions=4として固定値を設定)
    fixed_rewards = np.array([0.2, 0.15, 0.1, 0.05])

    # 文脈の数だけ期待報酬を繰り返して返す
    return np.tile(fixed_rewards, (n_rounds, 1))


def logging_policy_function_deterministic(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: int = None,
) -> np.ndarray:
    """(アクションa, 文脈x)の各組み合わせに対して、選択確率 π(a|x) を定義する関数。
    - 返り値のshape: (n_rounds, n_actions)
    - 今回は、決定論的に期待報酬の推定値 \hat{q}(x,a) が最大となるアクションを選択するデータ収集方策を設定
    - \hat{q}(x,a) は、今回はcontextによらず、事前に設定した固定の値とする
    - ニュース0: 0.05, ニュース1: 0.1, ニュース2: 0.15, ニュース3: 0.2
    - つまり任意の文脈xに対して、常にニュース4を選択するデータ収集方策を設定
    """
    n_rounds = context.shape[0]
    n_actions = action_context.shape[0]

    # 固定のスコアを設定 (n_actions=4として固定値を設定)
    fixed_scores = np.array([0.05, 0.1, 0.15, 0.2])

    # スコアが最大のアクションを確率1で選択する
    p_scores = np.array([0.0, 0.0, 0.0, 1.0])

    # 返り値の形式に整形: (n_rounds, n_actions)の配列で、各行が各ラウンドでのアクションの選択確率を表す
    action_dist = np.zeros((n_rounds, n_actions))
    action_dist[:, :] = p_scores

    assert np.allclose(
        action_dist.sum(axis=1), 1.0
    ), "各ラウンドでの全てのアクションの選択確率の合計は1.0である必要があります"

    return action_dist


def logging_policy_function_stochastic(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: int = None,
) -> np.ndarray:
    """(アクションa, 文脈x)の各組み合わせに対して、選択確率 π(a|x) を定義する関数
    - 今回は、確率的なデータ収集方策を設定する。
    - 返り値のshape: (n_rounds, n_actions)
    """
    n_rounds = context.shape[0]
    n_actions = action_context.shape[0]

    # 文脈xによらない固定の選択確率を設定する
    p_scores = np.array([0.1, 0.1, 0.1, 0.7])

    # 返り値の形式に整形: (n_rounds, n_actions)の配列で、各行が各ラウンドでのアクションの選択確率を表す
    action_dist = np.zeros((n_rounds, n_actions))
    action_dist[:, :] = p_scores

    assert np.allclose(
        action_dist.sum(axis=1), 1.0
    ), "各ラウンドでの全てのアクションの選択確率の合計は1.0である必要があります"
    return action_dist


# ニュース推薦用の文脈付きバンディットデータセットを生成
dataset = SyntheticBanditDataset(
    n_actions=4,
    dim_context=5,
    reward_type="binary",
    reward_function=expected_reward_function,
    # behavior_policy_function=logging_policy_function_deterministic,
    behavior_policy_function=logging_policy_function_stochastic,
    # random_state=123,
)

# バンディットデータを生成
bandit_feedback: dict = dataset.obtain_batch_bandit_feedback(n_rounds=10000)

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

# OPE推定量を準備(naive推定量と IPS推定量)
ope_estimators = [
    InverseProbabilityWeighting(),
    ReplayMethod(),
]

# オフライン評価用のクラスを初期化
ope = OffPolicyEvaluation(
    bandit_feedback=bandit_feedback,
    ope_estimators=ope_estimators,
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


# 評価方策を使って、bandit_feedbackに対する行動選択確率を計算
target_policy_action_dist = target_policy(
    context=bandit_feedback["context"],
    action_context=bandit_feedback["action_context"],
    recommend_arm_idx=1,
)

# 事前に設定した真の期待報酬E[r|x,a]を使って、評価方策の真の性能を計算
ground_truth_policy_value = dataset.calc_ground_truth_policy_value(
    expected_reward=bandit_feedback["expected_reward"],
    action_dist=target_policy_action_dist,
)
print(f"{ground_truth_policy_value=}")

# 各OPE推定量で、評価方策の性能を推定
estimated_policy_values = ope.estimate_policy_values(
    action_dist=target_policy_action_dist,
    estimated_pscore=bandit_feedback["pscore"],
)
print(f"{estimated_policy_values=}")
