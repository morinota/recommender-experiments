import numpy as np
from sklearn.neural_network import MLPRegressor
from recommender_experiments.service.synthetic_bandit_feedback import BanditFeedbackDict


def estimate_q_x_a_via_regression(
    bandit_feedback_train: BanditFeedbackDict,
    q_x_a_model=MLPRegressor(hidden_layer_sizes=(10, 10, 10), random_state=12345),
) -> np.ndarray:
    """DR推定量に使用する、期待報酬関数の予測モデル \hat{q}(x,a) を学習する関数
    Args:
        bandit_feedback_train (BanditFeedbackDict): 学習データ
        q_x_a_model (MLPRegressor, optional): 期待報酬関数の予測モデルのアーキテクチャ.
            Defaults to MLPRegressor(hidden_layer_sizes=(10, 10, 10), random_state=12345).
    Returns:
        np.ndarray: 各学習データに対する各アクションの期待報酬の予測値 \hat{q}(x,a) (shape: (n_rounds, n_actions))
    """
    n_data, n_actions = (
        bandit_feedback_train["n_rounds"],
        bandit_feedback_train["n_actions"],
    )
    x, r = bandit_feedback_train["context"], bandit_feedback_train["reward"]
    actions, a_feat = (
        bandit_feedback_train["action"],
        bandit_feedback_train["action_context"],
    )
    x_a = np.concatenate([x, a_feat[actions]], axis=1)

    # 学習データに対して、期待報酬関数の予測モデル \hat{q}(x,a) を学習
    q_x_a_model.fit(x_a, r)

    # 学習した \hat{q}(x,a) を用いて、学習データに対する各アクションの期待報酬の予測値を計算
    q_x_a_hat = np.zeros((n_data, n_actions))
    for a_idx in range(n_actions):
        x_a = np.concatenate([x, np.tile(a_feat[a_idx], (n_data, 1))], axis=1)
        q_x_a_hat[:, a_idx] = q_x_a_model.predict(x_a)

    return q_x_a_hat
