from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional
from obp.policy import NNPolicyLearner
import numpy as np
import torch

import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from recommender_experiments.service.estimator import estimate_q_x_a_via_regression
from recommender_experiments.service.synthetic_bandit_feedback import BanditFeedbackDict


@dataclass
class NNPolicyDataset(torch.utils.data.Dataset):
    """PyTorch dataset for NNPolicyLearner"""

    context: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    pscore: np.ndarray
    q_x_a_hat: np.ndarray
    pi_0: np.ndarray

    def __post_init__(self):
        """initialize class"""
        assert (
            self.context.shape[0]
            == self.action.shape[0]
            == self.reward.shape[0]
            == self.pscore.shape[0]
            == self.q_x_a_hat.shape[0]
            == self.pi_0.shape[0]
        )

    def __getitem__(self, index):
        return (
            self.context[index],
            self.action[index],
            self.reward[index],
            self.pscore[index],
            self.q_x_a_hat[index],
            self.pi_0[index],
        )

    def __len__(self):
        return self.context.shape[0]


@dataclass
class TwoTowerNNPolicyLearner:
    dim_context_features: int
    dim_action_features: int
    dim_two_tower_embedding: int
    softmax_temprature: float = 1.0
    hidden_layer_size: tuple = (30, 30, 30)
    activation: str = "elu"
    batch_size: int = 32
    learning_rate_init: float = 0.005
    alpha: float = 1e-6
    log_eps: float = 1e-10
    solver: str = "adam"
    max_iter: int = 200
    off_policy_objective: str = "ips"
    random_state: int = 12345

    def __post_init__(self):
        """Initialize class."""

        if self.activation == "tanh":
            activation_layer = nn.Tanh
        elif self.activation == "relu":
            activation_layer = nn.ReLU
        elif self.activation == "elu":
            activation_layer = nn.ELU

        # Context Tower
        context_tower_layers = []
        input_size = self.dim_context_features
        for idx, layer_size in enumerate(self.hidden_layer_size):
            context_tower_layers.append(
                (f"context_l_{idx}", nn.Linear(input_size, layer_size))
            )
            context_tower_layers.append((f"context_a_{idx}", activation_layer()))
            input_size = layer_size
        context_tower_layers.append(
            ("embed", nn.Linear(input_size, self.dim_two_tower_embedding))
        )
        self.context_tower = nn.Sequential(OrderedDict(context_tower_layers))

        # Action Tower
        action_layers = []
        input_size = self.dim_action_features
        for idx, layer_size in enumerate(self.hidden_layer_size):
            action_layers.append((f"action_l_{idx}", nn.Linear(input_size, layer_size)))
            action_layers.append((f"action_a_{idx}", activation_layer()))
            input_size = layer_size
        action_layers.append(
            ("embed", nn.Linear(input_size, self.dim_two_tower_embedding))
        )
        self.action_tower = nn.Sequential(OrderedDict(action_layers))

        self.nn_model = nn.ModuleDict(
            {
                "context_tower": self.context_tower,
                "action_tower": self.action_tower,
            }
        )

        self.train_losses = []
        self.train_values = []
        self.test_values = []

    def fit(
        self,
        # context: np.ndarray,  # context: (n_rounds, dim_context)
        # action: np.ndarray,  # action: (n_rounds, )
        # reward: np.ndarray,  # reward: (n_rounds,)
        # action_context: np.ndarray,  # action_context: (n_actions, dim_action_features)
        # pscore: Optional[np.ndarray] = None,  # pscore: (n_rounds,)
        # position: Optional[np.ndarray] = None,  # position: (n_rounds,)
        # q_x_a: Optional[np.ndarray] = None,  # q_x_a: (n_rounds, n_actions)
        bandit_feedback_train: BanditFeedbackDict,
        bandit_feedback_test: Optional[BanditFeedbackDict] = None,
    ) -> None:

        n_actions = bandit_feedback_train["n_actions"]
        context, action, reward, action_context, pscore, pi_b = (
            bandit_feedback_train["context"],
            bandit_feedback_train["action"],
            bandit_feedback_train["reward"],
            bandit_feedback_train["action_context"],
            bandit_feedback_train["pscore"],
            bandit_feedback_train["pi_b"],
        )

        # 期待報酬の推定モデル \hat{q}(x,a) を構築
        if self.off_policy_objective == "ips":
            q_x_a_hat = np.zeros((reward.shape[0], n_actions))
        elif self.off_policy_objective == "dr":
            q_x_a_hat = estimate_q_x_a_via_regression(bandit_feedback_train)

        # optimizerの設定
        if self.solver == "adagrad":
            optimizer = optim.Adagrad(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        elif self.solver == "adam":
            optimizer = optim.Adam(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        else:
            raise NotImplementedError("`solver` must be one of 'adam' or 'adagrad'")

        training_data_loader = self._create_train_data_for_opl(
            context,
            action,
            reward,
            pscore,
            q_x_a_hat,
            pi_b,
        )
        action_context_tensor = torch.from_numpy(action_context).float()

        # start policy training
        # n_not_improving_training = 0
        # previous_training_loss = None
        # n_not_improving_validation = 0
        # previous_validation_loss = None
        q_x_a_train = bandit_feedback_train["expected_reward"]
        q_x_a_test = bandit_feedback_test["expected_reward"]
        for _ in range(self.max_iter):
            # 各エポックの最初に、学習データとテストデータに対する真の方策性能を計算
            pi_train = self.predict_proba(
                context=context, action_context=action_context
            ).squeeze(-1)
            self.train_values.append((q_x_a_train * pi_train).sum(1).mean())
            pi_test = self.predict_proba(
                context=bandit_feedback_test["context"],
                action_context=bandit_feedback_test["action_context"],
            ).squeeze(-1)
            self.test_values.append((q_x_a_test * pi_test).sum(1).mean())

            loss_epoch = 0.0
            self.nn_model.train()
            for x, a, r, p, q_x_a_hat_, pi_b_ in training_data_loader:
                optimizer.zero_grad()
                # 新方策の行動選択確率分布\pi(a|x)を計算
                pi = self._predict_proba(
                    x, action_context_tensor
                )  # pi=(batch_size, n_actions)

                # 方策勾配の推定値を計算
                loss = -self._estimate_policy_gradient(
                    action=a,
                    reward=r,
                    pscore=p,
                    q_x_a_hat=q_x_a_hat_,
                    pi_0=pi_b_,
                    pi=pi,
                ).mean()
                # lossを最小化するようにモデルパラメータを更新
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()

            self.train_losses.append(loss_epoch)

        # 学習完了後に、学習データとテストデータに対する真の方策性能を計算
        pi_train = self.predict_proba(
            context=context, action_context=action_context
        ).squeeze(-1)
        self.train_values.append((q_x_a_train * pi_train).sum(1).mean())
        pi_test = self.predict_proba(
            context=bandit_feedback_test["context"],
            action_context=bandit_feedback_test["action_context"],
        ).squeeze(-1)
        self.test_values.append((q_x_a_test * pi_test).sum(1).mean())

    def _create_train_data_for_opl(
        self,
        context: np.ndarray,  # shape: (n_rounds, dim_context)
        action: np.ndarray,  # shape: (n_rounds,)
        reward: np.ndarray,  # shape: (n_rounds,)
        pscore: np.ndarray,  # shape: (n_rounds,)
        q_x_a_hat: np.ndarray,  # shape: (n_rounds, n_actions)
        pi_0: np.ndarray,  # shape: (n_rounds, n_actions)
        **kwargs,
    ) -> torch.utils.data.DataLoader:
        dataset = NNPolicyDataset(
            torch.from_numpy(context).float(),
            torch.from_numpy(action).long(),
            torch.from_numpy(reward).float(),
            torch.from_numpy(pscore).float(),
            torch.from_numpy(q_x_a_hat).float(),
            torch.from_numpy(pi_0).float(),
        )

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
        )
        return data_loader

    def _estimate_policy_gradient(
        self,
        action: torch.Tensor,  # shape: (batch_size,)
        reward: torch.Tensor,  # shape: (batch_size,)
        pscore: torch.Tensor,  # shape: (batch_size,)
        q_x_a_hat: torch.Tensor,  # shape: (batch_size, n_actions)
        pi: torch.Tensor,  # shape: (batch_size, n_actions, 1)
        pi_0: torch.Tensor,  # shape: (batch_size, n_actions)
    ) -> torch.Tensor:  # shape: (batch_size,)
        current_pi = pi.detach()
        log_prob = torch.log(pi + self.log_eps)
        idx_tensor = torch.arange(action.shape[0], dtype=torch.long)

        q_x_a_hat_factual = q_x_a_hat[idx_tensor, action]
        iw = current_pi[idx_tensor, action] / pscore
        estimated_policy_grad_arr = iw * (reward - q_x_a_hat_factual)
        estimated_policy_grad_arr *= log_prob[idx_tensor, action]
        estimated_policy_grad_arr += torch.sum(q_x_a_hat * current_pi * log_prob, dim=1)

        return estimated_policy_grad_arr

    def _predict_proba(
        self,
        context: torch.Tensor,  # shape: (n_rounds, dim_context_features)
        action_context: torch.Tensor,  # shape: (n_actions, dim_action_features)
    ) -> torch.Tensor:  # shape: (n_rounds, n_actions, 1)
        """方策による行動選択確率を予測するメソッド。
        行動選択確率は各アクションのロジット値を計算し、softmax関数を適用することで得られる。
        """

        # Context Tower Forward
        context_embedding = self.nn_model["context_tower"](
            context
        )  # shape: (n_rounds, dim_two_tower_embedding)
        context_embedding = context_embedding / context_embedding.norm(
            dim=-1, keepdim=True
        )

        # Action Tower Forward
        action_embedding = self.nn_model["action_tower"](
            action_context
        )  # shape: (n_actions, dim_two_tower_embedding)
        action_embedding = action_embedding / action_embedding.norm(
            dim=-1, keepdim=True
        )

        # context_embeddingとaction_embeddingの内積をスコアとして計算
        logits = torch.matmul(
            context_embedding, action_embedding.T
        )  # shape: (n_rounds, n_actions)

        # 行動選択確率分布を得るためにsoftmax関数を適用
        pi = torch.softmax(
            logits / self.softmax_temprature, dim=1
        )  # shape: (n_rounds, n_actions)

        return pi

    def predict_proba(
        self,
        context: np.ndarray,  # shape: (n_rounds, dim_context)
        action_context: np.ndarray,  # shape: (n_actions, dim_action_features)
    ) -> np.ndarray:  # shape: (n_rounds, n_actions, 1)
        """方策による行動選択確率を予測するメソッド"""
        assert context.shape[1] == self.dim_context_features
        assert action_context.shape[1] == self.dim_action_features

        self.nn_model.eval()

        action_dist = self._predict_proba(
            context=torch.from_numpy(context).float(),
            action_context=torch.from_numpy(action_context).float(),
        )
        action_dist_ndarray = action_dist.squeeze(-1).detach().numpy()
        return action_dist_ndarray[:, :, np.newaxis]  # shape: (n_rounds, n_actions, 1)


if __name__ == "__main__":
    # Arrange
    n_rounds = 20000
    n_actions = 4
    dim_context = 300
    dim_action_features = 200
    dim_two_tower_embedding = 120

    sut = TwoTowerNNPolicyLearner(
        dim_context=dim_context,
        dim_action_features=dim_action_features,
        dim_two_tower_embedding=dim_two_tower_embedding,
        is_embedding_normed=True,
        batch_size=20000,
    )

    # Act
    sut.fit(
        context=np.random.random((n_rounds, dim_context)),
        action=np.random.randint(0, n_actions, n_rounds),
        reward=np.random.binomial(1, 0.5, n_rounds),
        action_context=np.random.random((n_actions, dim_action_features)),
    )
    action_dist = sut.predict_proba(
        context=np.random.random((n_rounds, dim_context)),
        action_context=np.random.random((n_actions, dim_action_features)),
    )

    # Assert
    assert action_dist.shape == (
        n_rounds,
        n_actions,
        1,
    ), "各ラウンドごとに、確率の総和が1.0"
    assert np.all(0 <= action_dist) and np.all(
        action_dist <= 1
    ), "各アクションの選択確率が0以上1以下であること"
