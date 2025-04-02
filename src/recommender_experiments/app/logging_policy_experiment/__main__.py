import random
from typing import Callable, Optional, TypedDict
import numpy as np
from obp.dataset import SyntheticBanditDataset
from obp.ope import ReplayMethod, InverseProbabilityWeighting
import polars as pl


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


def expected_reward_function(context: np.ndarray, action_context: np.ndarray, random_state: int = None) -> np.ndarray:
    """(アクションa, 文脈x)の各組み合わせに対する期待報酬 E_{p(r|x,a)}[r] を定義する関数
    今回の場合は、推薦候補4つの記事を送った場合の報酬rの期待値を、文脈xに依存しない固定値として設定する
    ニュース0: 0.2, ニュース1: 0.15, ニュース2: 0.1, ニュース3: 0.05
    返り値のshape: (n_rounds, n_actions, len_list)å
    """
    n_rounds = context.shape[0]
    n_actions = action_context.shape[0]

    # 固定の期待報酬を設定 (n_actions=4として固定値を設定)
    fixed_rewards = np.array([0.4, 0.3, 0.2, 0.2])

    # 文脈の数だけ期待報酬を繰り返して返す
    return np.tile(fixed_rewards, (n_rounds, 1))


def logging_policy_function_deterministic(
    context: np.ndarray, action_context: np.ndarray, random_state: int = None
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

    assert np.allclose(action_dist.sum(axis=1), 1.0), (
        "各ラウンドでの全てのアクションの選択確率の合計は1.0である必要があります"
    )

    return action_dist


def logging_policy_function_stochastic(
    context: np.ndarray, action_context: np.ndarray, random_state: int = None
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

    assert np.allclose(action_dist.sum(axis=1), 1.0), (
        "各ラウンドでの全てのアクションの選択確率の合計は1.0である必要があります"
    )
    return action_dist


# 評価方策(target policy)として、常にニュース0を選択する方策を設定
def target_policy_1(
    context: np.ndarray, action_context: np.ndarray, random_state: int = None, recommend_arm_idx: int = 0
) -> np.ndarray:  # shape: (n_rounds, n_actions, len_list)
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

    assert np.allclose(action_dist.sum(axis=1), 1.0), (
        "各ラウンドでの全てのアクションの選択確率の合計は1.0である必要があります"
    )

    return action_dist


def target_policy_2(
    context: np.ndarray,  # shape: (n_rounds, dim_context)
    action_context: np.ndarray,  # shape: (n_actions, dim_action_features)
    random_state: int = None,
) -> np.ndarray:  # shape: (n_rounds, n_actions, len_list)
    """contextとaction_contextを同じベクトル空間のベクトルとみなし、contextとaction_contextの内積が最大となるアクションを選択する方策"""
    n_rounds = context.shape[0]
    n_actions = action_context.shape[0]

    # 各ラウンドでcontextとaction_contextの内積が最大となるアクションを選択
    dot_products = context @ action_context.T
    selected_action_indices = np.argmax(dot_products, axis=1)

    # recommend_arm_idx番目のアクションの選択確率を1.0、それ以外のアクションの選択確率を0.0とする
    p_scores = np.zeros((n_rounds, n_actions))
    p_scores[np.arange(n_rounds), selected_action_indices] = 1.0

    # 返り値の形式に整形: (n_rounds, n_actions, 1)の配列で、各行が各ラウンドでのアクションの選択確率を表す
    action_dist = np.zeros((n_rounds, n_actions, 1))
    action_dist[:, :, 0] = p_scores

    assert np.allclose(action_dist.sum(axis=1), 1.0), (
        "各ラウンドでの全てのアクションの選択確率の合計は1.0である必要があります"
    )

    return action_dist


def run_single_simulation(
    simulate_idx: int,
    n_rounds: int,
    n_actions: int,
    dim_context: int,
    reward_type: str,
    reward_function: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
    logging_policy_function: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
    action_context: np.ndarray,
) -> dict:
    # データ収集方策によって集められるはずの、擬似バンディットデータの設定を定義
    dataset = SyntheticBanditDataset(
        n_actions=n_actions,
        dim_context=dim_context,
        reward_type=reward_type,
        reward_function=reward_function,
        behavior_policy_function=logging_policy_function,
        random_state=simulate_idx,
        action_context=action_context,
    )
    # 収集されるバンディットフィードバックデータを生成
    bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)

    # bandit_feedbackをpl.DataFrameに変換
    selected_action_contexts = action_context[bandit_feedback["action"], :]
    bandit_feedback_df = pl.DataFrame(
        {
            "time_step": [i for i in range(n_rounds)],
            "context": bandit_feedback["context"].tolist(),
            "action": bandit_feedback["action"].tolist(),
            "action_context": selected_action_contexts.tolist(),
            "reward": bandit_feedback["reward"].tolist(),
            "p_score": bandit_feedback["pscore"].tolist(),
        }
    )
    print(bandit_feedback_df)

    # 評価方策を使って、ログデータ(bandit_feedback)に対する行動選択確率を計算
    # target_policy_action_dist = target_policy_1(
    #     context=bandit_feedback["context"],
    #     action_context=bandit_feedback["action_context"],
    #     recommend_arm_idx=0,
    # )
    target_policy_action_dist = target_policy_2(
        context=bandit_feedback["context"], action_context=bandit_feedback["action_context"]
    )
    # 真の期待報酬E[r|x,a]を使って、データ収集方策の代わりに評価方策を動かした場合の価値を算出
    ground_truth_policy_value = dataset.calc_ground_truth_policy_value(
        expected_reward=bandit_feedback["expected_reward"], action_dist=target_policy_action_dist
    )

    # OPE推定量を準備(naive推定量とIPS推定量)
    naive_estimator = ReplayMethod()
    ips_estimator = InverseProbabilityWeighting()

    # それぞれのOPE推定量を使って、データ収集方策の代わりに評価方策を動かした場合の価値を推定
    estimated_policy_value_by_naive = naive_estimator.estimate_policy_value(
        reward=bandit_feedback["reward"], action=bandit_feedback["action"], action_dist=target_policy_action_dist
    )
    estimated_policy_value_by_ips = ips_estimator.estimate_policy_value(
        action=bandit_feedback["action"],
        reward=bandit_feedback["reward"],
        action_dist=target_policy_action_dist,
        pscore=bandit_feedback["pscore"],
    )

    return {
        "simulate_idx": simulate_idx,
        "ground_truth_policy_value": ground_truth_policy_value,
        "estimated_policy_value_by_naive": estimated_policy_value_by_naive,
        "estimated_policy_value_by_ips": estimated_policy_value_by_ips,
    }


def main() -> None:
    # シミュレーションの設定
    n_sim = 1
    n_rounds = 10000
    n_actions = 4
    dim_context = 2
    reward_type = "binary"
    # シードを固定
    random.seed(12345)
    action_context = np.random.randn(n_actions, dim_context)

    results = []
    for simulate_idx in range(n_sim):
        simulate_result = run_single_simulation(
            simulate_idx=simulate_idx,
            n_rounds=n_rounds,
            n_actions=n_actions,
            dim_context=dim_context,
            reward_type=reward_type,
            reward_function=expected_reward_function,
            logging_policy_function=logging_policy_function_stochastic,
            action_context=action_context,
        )

        # 結果の表示
        print(simulate_result)
        results.append(simulate_result)

    # シミュレーション結果を集計して表示
    result_df = pl.DataFrame(results)
    print(result_df)


if __name__ == "__main__":
    main()
