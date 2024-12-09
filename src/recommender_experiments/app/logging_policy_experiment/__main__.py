import numpy as np
from obp.dataset import logistic_reward_function, SyntheticBanditDataset


def expected_reward_function(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: int = None,  # この引数を追加
) -> np.ndarray:
    """(アクションa, 文脈x)の各組み合わせに対する期待報酬 E[r|x,a] を定義する関数
    今回の場合は、推薦候補4つの記事を送った場合の報酬rの期待値を、文脈xに依存しない固定値として設定する
    ニュース1: 0.2, ニュース2: 0.15, ニュース3: 0.1, ニュース4: 0.05
    """
    n_rounds = context.shape[0]
    n_actions = action_context.shape[0]

    # 固定の期待報酬を設定 (n_actions=4として固定値を設定)
    fixed_rewards = np.array([0.2, 0.15, 0.1, 0.05])

    # 文脈の数だけ期待報酬を繰り返して返す
    return np.tile(fixed_rewards, (n_rounds, 1))


def logging_policy_function_ver1(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: int = None,  # この引数を追加
) -> np.ndarray:
    """(アクションa, 文脈x)の各組み合わせに対して、選択確率 π(a|x) を定義する関数
    - 今回は、決定論的に期待報酬の推定値 \hat{q}(x,a) が最大となるアクションを選択するデータ収集方策を設定
    - \hat{q}(x,a) は、今回はcontextによらず、事前に設定した固定の値とする
    - ニュース0: 0.05, ニュース1: 0.1, ニュース2: 0.15, ニュース3: 0.2
    - つまり任意の文脈xに対して、常にニュース4を選択するデータ収集方策を設定
    """
    n_rounds = context.shape[0]
    n_actions = action_context.shape[0]

    # 固定の選択確率を設定 (n_actions=4として固定値を設定)
    fixed_ps = np.array([0.05, 0.1, 0.15, 0.2])

    # スコアが最大となるアクションを確率1で選択する
    action_dist = np.zeros((n_rounds, n_actions))
    action_dist[:, np.argmax(fixed_ps)] = 1.0

    return action_dist


# ニュース推薦用の文脈付きバンディットデータセットを生成
dataset = SyntheticBanditDataset(
    n_actions=4,  # ニュース記事（アクション）の数
    dim_context=5,  # 文脈ベクトルの次元数
    reward_type="binary",  # 報酬となるmetricsの種類("binary" or "continuous")
    reward_function=expected_reward_function,  # 期待報酬関数
    behavior_policy_function=logging_policy_function_ver1,
    random_state=123,  # 再現性のためのランダムシード
)

# バンディットデータを生成
bandit_feedback: dict = dataset.obtain_batch_bandit_feedback(n_rounds=5)

# バンディットデータの中身を確認
print(f"{bandit_feedback.keys()=}")
print(f"{bandit_feedback['n_rounds']=}")
print(f"{bandit_feedback['n_actions']=}")
print(f"{bandit_feedback['action']=}")
print(f"{bandit_feedback['position']=}")
print(f"{bandit_feedback['reward']=}")
print(f"{bandit_feedback['expected_reward']=}")
print(f"{bandit_feedback['pi_b']=}")
print(f"{bandit_feedback['pscore']=}")
