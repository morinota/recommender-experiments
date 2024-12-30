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
class TwoTowerNNPolicyLearner(NNPolicyLearner):
    n_actions: int = (
        None  # dummyのパラメータ(アクション数がdynamicに変化する前提なので)
    )
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
    dim_action_features: int = None
    dim_two_tower_embedding: int = None

    def __post_init__(self):
        print("overriden __post_init__ for TwoTower Model")

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

        # Context Tower
        context_layers = []
        input_size = self.dim_context

        for i, h in enumerate(self.hidden_layer_size):
            print(f"Context Tower - Layer {i}: {input_size} -> {h}")
            context_layers.append((f"context_l{i}", nn.Linear(input_size, h)))
            context_layers.append((f"context_a{i}", activation_layer()))
            input_size = h
        context_layers.append(
            ("embed", nn.Linear(input_size, self.dim_two_tower_embedding))
        )
        self.context_tower = nn.Sequential(OrderedDict(context_layers))

        # Action Tower
        action_layers = []
        input_size = self.dim_action_features

        for i, h in enumerate(self.hidden_layer_size):
            print(f"Action Tower - Layer {i}: {input_size} -> {h}")
            action_layers.append((f"action_l{i}", nn.Linear(input_size, h)))
            action_layers.append((f"action_a{i}", activation_layer()))
            input_size = h
        action_layers.append(
            ("embed", nn.Linear(input_size, self.dim_two_tower_embedding))
        )
        self.action_tower = nn.Sequential(OrderedDict(action_layers))

        # Output Layer
        self.output_layer = nn.Linear(self.dim_two_tower_embedding * 2, 1)

        # Combine all layers
        self.nn_model = nn.ModuleDict(
            {
                "context_tower": self.context_tower,
                "action_tower": self.action_tower,
                "output_layer": self.output_layer,
            }
        )

    def _predict_proba_for_fit(
        self,
        context: torch.Tensor,  # shape: (n_rounds, dim_context)
        action_context: torch.Tensor,  # shape: (n_actions, dim_action_features)
    ) -> np.ndarray:  # shape: (n_rounds, n_actions, 1)
        self.context_tower.eval()
        self.action_tower.eval()

        n_rounds = context.shape[0]
        n_actions = action_context.shape[0]

        # Context Tower Forward
        context_embedding = self.context_tower(
            context
        )  # shape: (n_rounds, hidden_size)

        # Action Tower Forward
        action_embedding = self.action_tower(
            action_context
        )  # shape: (n_actions, hidden_size)

        # Combine context and action embeddings
        combined_embedding = torch.cat(
            [
                context_embedding.unsqueeze(1).expand(
                    -1, n_actions, -1
                ),  # (n_rounds, n_actions, hidden_size)
                action_embedding.unsqueeze(0).expand(
                    n_rounds, -1, -1
                ),  # (n_rounds, n_actions, hidden_size)
            ],
            dim=2,
        )  # shape: (n_rounds, n_actions, hidden_size * 2)

        # Pass through output layer
        combined_embedding = combined_embedding.view(-1, combined_embedding.size(2))
        scores = self.output_layer(combined_embedding).view(
            n_rounds, n_actions
        )  # shape: (n_rounds, n_actions)

        # Softmax to get probabilities
        pi = torch.softmax(scores, dim=1)  # shape: (n_rounds, n_actions)

        return pi.unsqueeze(-1)

    # predict_probaメソッドをoverride
    def predict_proba(
        self,
        context: np.ndarray,  # shape: (n_rounds, dim_context)
        action_context: np.ndarray,  # shape: (n_actions, dim_action_features)
    ) -> np.ndarray:
        """方策による行動選択確率を予測するメソッド"""
        context_tensor = torch.from_numpy(context).float()
        action_context_tensor = torch.from_numpy(action_context).float()
        pi = self._predict_proba_for_fit(
            context=context_tensor,
            action_context=action_context_tensor,
        )
        return pi.squeeze(-1).detach().numpy()

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
                    context=x,
                    reward=r,
                    action=a,
                    pscore=p,
                    action_dist=pi,
                    position=pos,
                )
                policy_constraint = self._estimate_policy_constraint(
                    action=a,
                    pscore=p,
                    action_dist=pi,
                )
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


if __name__ == "__main__":
    # 基本設定
    n_rounds = 10
    n_actions = 4
    dim_context = 300
    dim_action_features = 200
    dim_two_tower_embedding = 100
    # ダミーデータの生成
    context = np.random.random((n_rounds, dim_context))
    action = np.random.randint(0, n_actions, n_rounds)
    reward = np.random.binomial(1, 0.5, n_rounds)  # binaryのrewardを生成
    action_context = np.random.random((n_actions, dim_action_features))

    # two-tower modelに対して、オフ方策学習を実行
    policy = TwoTowerNNPolicyLearner(
        dim_context=dim_context,
        dim_action_features=dim_action_features,
        dim_two_tower_embedding=dim_two_tower_embedding,
    )
    # バンディットフィードバックデータを用いてオフ方策学習
    policy.fit(context, action, reward, action_context)
    # 方策による行動選択確率の出力
    action_dist = policy.predict_proba(context, action_context)
    print(f"{action_dist=}")

    # アクション数が動的に変化しても、同一モデルで推論できることを確認
    n_actions += 2
    action_context_ver2 = np.random.random((n_actions, dim_action_features))
    action_dist_ver2 = policy.predict_proba(context, action_context_ver2)
    print(f"{action_dist_ver2=}")
