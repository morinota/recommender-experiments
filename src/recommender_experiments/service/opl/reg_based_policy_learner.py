from dataclasses import dataclass
import numpy as np
from sklearn.neural_network import MLPRegressor
from recommender_experiments.service.synthetic_bandit_feedback import BanditFeedbackDict


def softmax(x: np.ndarray) -> np.ndarray:
    """確率的方策を定義するときに使う、softmax関数"""
    # オーバーフローを避けるために、入力信号の最大値を引く(出力は不変)
    b = np.max(x, axis=1)[:, np.newaxis]
    numerator = np.exp(x - b)
    denominator = np.sum(numerator, axis=1)[:, np.newaxis]
    return numerator / denominator


@dataclass
class RegBasedPolicyLearner:
    """回帰ベースのオフ方策学習器
    実装参考:https://github.com/usaito/www2024-lope/blob/main/notebooks/learning.py
    """

    q_x_a_model = MLPRegressor(hidden_layer_sizes=(30, 30, 30), random_state=12345)
    random_state: int = 12345

    def fit(self, bandit_feedback: BanditFeedbackDict) -> None:
        x, r = bandit_feedback["context"], bandit_feedback["reward"]
        actions, a_feat = bandit_feedback["action"], bandit_feedback["action_context"]

        x_a = np.concatenate([x, a_feat[actions]], 1)
        self.q_x_a_model.fit(x_a, r)

    def predict(self, D_test: BanditFeedbackDict, tau: float = 0.01) -> np.ndarray:
        n_data, n_actions = D_test["n_rounds"], D_test["n_actions"]
        x, a_feat = D_test["context"], D_test["action_context"]
        q_x_a_hat = np.zeros((n_data, n_actions))
        for a in range(n_actions):
            x_a = np.concatenate([x, np.tile(a_feat[a], (n_data, 1))], 1)
            q_x_a_hat[:, a] = self.q_x_a_model.predict(x_a)

        return softmax(q_x_a_hat / tau)
