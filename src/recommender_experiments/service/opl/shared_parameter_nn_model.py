from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional
from obp.policy import NNPolicyLearner
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


@dataclass
class SharedParameterNNPolicyLearner(NNPolicyLearner):
    n_actions: int = None  # dummyのパラメータ(アクション数がdynamicに変化する前提なので)
    dim_context: int
    off_policy_objective: str = "ipw"
    lambda_: Optional[float] = None
    policy_reg_param: float = 0.0
    var_reg_param: float = 0.0
    hidden_layer_size: tuple[int, ...] = (100,)
    activation: str = "relu"
    solver: str = "adam"
    alpha: float = 0.0001
    batch_size: tuple[int, str] = "auto"
    learning_rate_init: float = 0.0001
    max_iter: int = 200
    shuffle: bool = True
    random_state: Optional[int] = None
    tol: float = 1e-4
    momentum: float = 0.9
    nesterovs_momentum: bool = True
    early_stopping: bool = False
    validation_fraction: float = 0.1
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-8
    n_iter_no_change: int = 10
    q_func_estimator_hyperparams: Optional[dict] = None

    # __post_init__をoverride
    def __post_init__(self):
        print("overriden __post_init__")

        if self.activation == "identity":
            activation_layer = nn.Identity
        elif self.activation == "logistic":
            activation_layer = nn.Sigmoid
        elif self.activation == "tanh":
            activation_layer = nn.Tanh
        elif self.activation == "relu":
            activation_layer = nn.ReLU
        elif self.activation == "elu":
            activation_layer = nn.ELU
        else:
            raise ValueError(
                "`activation` must be one of 'identity', 'logistic', 'tanh', 'relu', or 'elu'"
                f", but {self.activation} is given"
            )

        layer_list = []
        input_size = self.dim_context

        for i, h in enumerate(self.hidden_layer_size):
            print(f"i: {i}, h: {h}")
            layer_list.append((f"l{i}", nn.Linear(input_size, h)))
            layer_list.append((f"a{i}", activation_layer()))
            input_size = h
        layer_list.append(("output", nn.Linear(input_size, 1)))
        layer_list.append(("sigmoid", nn.Sigmoid()))

        self.nn_model = nn.Sequential(OrderedDict(layer_list))

    # fitメソッドをoverride
    def fit(
        self,
        context: np.ndarray,  # context: (n_rounds, dim_context)
        action: np.ndarray,  # action: (n_rounds, )
        reward: np.ndarray,  # reward: (n_rounds,)
        action_context: np.ndarray,  # action_context: (n_actions, dim_action_features)
        pscore: Optional[np.ndarray] = None,  # pscore: (n_rounds,)
        position: Optional[np.ndarray] = None,  # position: (n_rounds,)
    ):
        print("overriden fit method")
        self.n_actions = action_context.shape[0]  # ここでアクション数を上書き

        # 入力データのvalidation & parse
        if pscore is None:
            pscore = np.ones_like(action) / self.n_actions
        if self.len_list == 1:
            position = np.zeros_like(action, dtype=int)

        # optimizerの設定
        optimizer = optim.Adam(self.nn_model.parameters(), lr=self.learning_rate_init)

        training_data_loader, validation_data_loader = self._create_train_data_for_opl(
            context, action, reward, pscore, position
        )
        action_context_tensor = torch.from_numpy(action_context).float()

        # start policy training
        n_not_improving_training = 0
        previous_training_loss = None
        n_not_improving_validation = 0
        previous_validation_loss = None
        for _ in tqdm(range(self.max_iter), desc="policy learning"):
            self.nn_model.train()
            for x, a, r, p, pos in training_data_loader:
                optimizer.zero_grad()
                # 新方策の行動選択確率分布\pi(a|x)を計算
                pi = self._predict_proba_for_fit(x, action_context_tensor)

                # 方策勾配の推定値を計算
                policy_grad_arr = self._estimate_policy_gradient(
                    context=x, reward=r, action=a, pscore=p, action_dist=pi, position=pos
                )
                policy_constraint = self._estimate_policy_constraint(action=a, pscore=p, action_dist=pi)
                loss = -policy_grad_arr.mean()
                loss += self.policy_reg_param * policy_constraint
                loss += self.var_reg_param * torch.var(policy_grad_arr)
                loss.backward()
                optimizer.step()

                # パラメータ更新後の損失を計算
                loss_value = loss.item()
                if previous_training_loss is not None:
                    if loss_value - previous_training_loss < self.tol:
                        n_not_improving_training += 1
                    else:
                        n_not_improving_training = 0
                if n_not_improving_training >= self.n_iter_no_change:
                    break
                previous_training_loss = loss_value

    def _predict_proba_for_fit(
        self,
        context: torch.Tensor,  # shape: (n_rounds, dim_context)
        action_context: torch.Tensor,  # shape: (n_actions, dim_action_features)
    ) -> np.ndarray:  # shape: (n_rounds, n_actions, 1)
        n_rounds = context.shape[0]
        n_actions = action_context.shape[0]

        # contextを複製してaction_contextと結合
        combined_input = torch.cat(
            [
                context.unsqueeze(1).expand(-1, n_actions, -1),  # shape: (n_rounds, n_actions, dim_context)
                action_context.unsqueeze(0).expand(
                    n_rounds, -1, -1
                ),  # shape: (n_rounds, n_actions, dim_action_features)
            ],
            dim=2,
        )  # shape: (n_rounds, n_actions, dim_context + dim_action_features)

        # 2次元に変形してNNに入力
        combined_input = combined_input.view(
            -1, combined_input.size(2)
        )  # shape: (n_rounds * n_actions, dim_context + dim_action_features)
        scores = self.nn_model(combined_input).view(n_rounds, n_actions)  # shape: (n_rounds, n_actions)

        # 各ラウンドごとに、アクション選択確率の総和が1.0になるようにsoftmax関数を適用
        pi = torch.softmax(scores, dim=1)  # shape: (n_rounds, n_actions)

        return pi.unsqueeze(-1)  # shape: (n_rounds, n_actions, 1)

    # predict_probaメソッドをoverride
    def predict_proba(
        self,
        context: np.ndarray,  # shape: (n_rounds, dim_context)
        action_context: np.ndarray,  # shape: (n_actions, dim_action_features)
    ) -> np.ndarray:
        action_context_tensor = torch.from_numpy(action_context).float()
        pi = self._predict_proba_for_fit(
            context=torch.from_numpy(context).float(),  # shape: (n_rounds, dim_context)
            action_context=action_context_tensor,  # shape: (n_actions, dim_action_features)
        )
        action_dist_ndarray = pi.squeeze(-1).detach().numpy()
        return action_dist_ndarray[:, :, np.newaxis]  # shape: (n_rounds, n_actions, 1)
