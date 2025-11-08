from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import check_random_state
from torch.optim.lr_scheduler import ExponentialLR

from recommender_experiments.app.cfml_book.ch5.utils import GradientBasedPolicyDataset


@dataclass
class GradientBasedPolicyLearner:
    """勾配ベースのアプローチに基づくオフ方策学習"""

    dim_x: int
    num_actions: int
    hidden_layer_size: tuple = (30, 30, 30)
    activation: str = "elu"
    batch_size: int = 16
    learning_rate_init: float = 0.005
    gamma: float = 0.98
    alpha: float = 1e-6
    imit_reg: float = 0.0
    log_eps: float = 1e-10
    solver: str = "adagrad"
    max_iter: int = 30
    random_state: int = 12345

    def __post_init__(self) -> None:
        """Initialize class."""
        layer_list = []
        input_size = self.dim_x

        if self.activation == "tanh":
            activation_layer = nn.Tanh
        elif self.activation == "relu":
            activation_layer = nn.ReLU
        elif self.activation == "elu":
            activation_layer = nn.ELU

        for i, h in enumerate(self.hidden_layer_size):
            layer_list.append(("l{}".format(i), nn.Linear(input_size, h)))
            layer_list.append(("a{}".format(i), activation_layer()))
            input_size = h
        layer_list.append(("output", nn.Linear(input_size, self.num_actions)))
        layer_list.append(("softmax", nn.Softmax(dim=1)))

        self.nn_model = nn.Sequential(OrderedDict(layer_list))

        self.random_ = check_random_state(self.random_state)
        self.train_loss = []
        self.train_value = []
        self.test_value = []

    def fit(self, dataset: dict, dataset_test: dict, q_hat: np.ndarray = None) -> None:
        x, a, r = dataset["x"], dataset["a"], dataset["r"]
        pscore, pi_0 = dataset["pscore"], dataset["pi_0"]
        if q_hat is None:
            q_hat = np.zeros((r.shape[0], self.num_actions))

        if self.solver == "adagrad":
            optimizer = optim.Adagrad(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        elif self.solver == "adam":
            optimizer = optim.AdamW(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        else:
            raise NotImplementedError("`solver` must be one of 'adam' or 'adagrad'")

        training_data_loader = self._create_train_data_for_opl(
            x,
            a,
            r,
            pscore,
            q_hat,
            pi_0,
        )

        # start policy training
        scheduler = ExponentialLR(optimizer, gamma=self.gamma)
        q_x_a_train, q_x_a_test = dataset["q_x_a"], dataset_test["q_x_a"]
        for _ in range(self.max_iter):
            loss_epoch = 0.0
            self.nn_model.train()
            for x_, a_, r_, p, q_hat_, pi_0_ in training_data_loader:
                optimizer.zero_grad()
                pi = self.nn_model(x_)
                loss = -self._estimate_policy_gradient(
                    a=a_,
                    r=r_,
                    pscore=p,
                    q_hat=q_hat_,
                    pi_0=pi_0_,
                    pi=pi,
                ).mean()
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
            self.train_loss.append(loss_epoch)
            scheduler.step()
            pi_train = self.predict(dataset)
            self.train_value.append((q_x_a_train * pi_train).sum(1).mean())
            pi_test = self.predict(dataset_test)
            self.test_value.append((q_x_a_test * pi_test).sum(1).mean())

    def _create_train_data_for_opl(
        self,
        x: np.ndarray,
        a: np.ndarray,
        r: np.ndarray,
        pscore: np.ndarray,
        q_hat: np.ndarray,
        pi_0: np.ndarray,
    ) -> tuple:
        dataset = GradientBasedPolicyDataset(
            torch.from_numpy(x).float(),
            torch.from_numpy(a).long(),
            torch.from_numpy(r).float(),
            torch.from_numpy(pscore).float(),
            torch.from_numpy(q_hat).float(),
            torch.from_numpy(pi_0).float(),
        )

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
        )

        return data_loader

    def _estimate_policy_gradient(
        self,
        a: torch.Tensor,
        r: torch.Tensor,
        pscore: torch.Tensor,
        q_hat: torch.Tensor,
        pi: torch.Tensor,
        pi_0: torch.Tensor,
    ) -> torch.Tensor:
        current_pi = pi.detach()
        log_prob = torch.log(pi + self.log_eps)
        idx = torch.arange(a.shape[0], dtype=torch.long)

        q_hat_factual = q_hat[idx, a]
        iw = current_pi[idx, a] / pscore
        estimated_policy_grad_arr = iw * (r - q_hat_factual) * log_prob[idx, a]
        estimated_policy_grad_arr += torch.sum(q_hat * current_pi * log_prob, dim=1)

        # imitation regularization
        estimated_policy_grad_arr += self.imit_reg * log_prob[idx, a]

        return estimated_policy_grad_arr

    def predict(self, dataset_test: np.ndarray) -> np.ndarray:
        self.nn_model.eval()
        x = torch.from_numpy(dataset_test["x"]).float()
        return self.nn_model(x).detach().numpy()
