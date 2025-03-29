from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch

import torch.nn as nn
import torch.optim as optim

from recommender_experiments.service.estimator import estimate_q_x_a_via_regression
from recommender_experiments.service.opl.policy_strategy_interface import (
    PolicyStrategyInterface,
)
from recommender_experiments.service.synthetic_bandit_feedback import BanditFeedbackDict


@dataclass
class NNPolicyDataset(torch.utils.data.Dataset):
    """Two-Towerモデルのオフ方策学習用のデータセットクラス"""

    context: np.ndarray  # 文脈x_i
    action: np.ndarray  # 行動a_i
    reward: np.ndarray  # 報酬r_i
    pscore: np.ndarray  # 傾向スコア \pi_0(a_i|x_i)
    q_x_a_hat: np.ndarray  # 期待報酬の推定値 \hat{q}(x_i, a)

    def __post_init__(self):
        """initialize class"""
        assert (
            self.context.shape[0]
            == self.action.shape[0]
            == self.reward.shape[0]
            == self.pscore.shape[0]
            == self.q_x_a_hat.shape[0]
        )

    def __getitem__(self, index):
        return (
            self.context[index],
            self.action[index],
            self.reward[index],
            self.pscore[index],
            self.q_x_a_hat[index],
        )

    def __len__(self):
        return self.context.shape[0]


@dataclass
class PolicyByTwoTowerModel(PolicyStrategyInterface):
    """Two-Towerモデルのオフ方策学習を行うクラス
    実装参考: https://github.com/usaito/www2024-lope/blob/main/notebooks/learning.py
    """

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
    off_policy_objective: str = "dr"
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

    @property
    def policy_name(self) -> str:
        return "two-tower-policy"

    def fit(
        self,
        bandit_feedback_train: BanditFeedbackDict,
        bandit_feedback_test: Optional[BanditFeedbackDict] = None,
    ) -> None:
        """推薦方策を学習するメソッド"""
        if self.off_policy_objective in ("ips", "dr"):
            self._fit_by_gradiant_based_approach(
                bandit_feedback_train=bandit_feedback_train,
                bandit_feedback_test=bandit_feedback_test,
            )
        elif self.off_policy_objective == "regression_based":
            self._fit_by_regression_based_approach(
                bandit_feedback_train=bandit_feedback_train,
                bandit_feedback_test=bandit_feedback_test,
            )
        else:
            raise NotImplementedError(
                "`off_policy_objective` must be one of 'ips', 'dr', or 'regression_based'"
            )

    def _fit_by_gradiant_based_approach(
        self,
        bandit_feedback_train: BanditFeedbackDict,
        bandit_feedback_test: Optional[BanditFeedbackDict] = None,
    ) -> None:
        """推薦方策を、勾配ベースアプローチで学習するメソッド"""

        n_actions = bandit_feedback_train["n_actions"]
        context, action, reward, action_context, pscore, pi_b = (
            bandit_feedback_train["context"],
            bandit_feedback_train["action"],
            bandit_feedback_train["reward"],
            bandit_feedback_train["action_context"],
            bandit_feedback_train["pscore"],
            bandit_feedback_train["pi_b"],
        )

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

        # 期待報酬の推定モデル \hat{q}(x,a) を構築
        if self.off_policy_objective == "ips":
            q_x_a_hat = np.zeros((reward.shape[0], n_actions))
        elif self.off_policy_objective == "dr":
            q_x_a_hat = estimate_q_x_a_via_regression(bandit_feedback_train)
        else:
            raise NotImplementedError

        training_data_loader = self._create_train_data_for_opl(
            context,
            action,
            reward,
            pscore,
            q_x_a_hat,
        )
        action_context_tensor = torch.from_numpy(action_context).float()

        # start policy training
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
            for x, a, r, p, q_x_a_hat_ in training_data_loader:
                optimizer.zero_grad()
                # 新方策の行動選択確率分布\pi(a|x)を計算
                pi = self._predict_proba_as_tensor(
                    x, action_context_tensor
                )  # pi=(batch_size, n_actions)

                # 方策勾配の推定値を計算 (方策性能を最大化したいのでマイナスをかけてlossとする)
                loss = -self._estimate_policy_gradient(
                    action=a,
                    reward=r,
                    pscore=p,
                    q_x_a_hat=q_x_a_hat_,
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
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        q_x_a_hat: np.ndarray,
        **kwargs,
    ) -> torch.utils.data.DataLoader:
        """学習データを作成するメソッド
        Args:
            context (np.ndarray): コンテキスト特徴量の配列 (n_rounds, dim_context_features)
            action (np.ndarray): 選択されたアクションの配列 (n_rounds,)
            reward (np.ndarray): 観測された報酬の配列 (n_rounds,)
            pscore (np.ndarray): 傾向スコアの配列 (n_rounds,)
            q_x_a_hat (np.ndarray): 期待報酬の推定値の配列 (n_rounds, n_actions)
        """
        dataset = NNPolicyDataset(
            torch.from_numpy(context).float(),
            torch.from_numpy(action).long(),
            torch.from_numpy(reward).float(),
            torch.from_numpy(pscore).float(),
            torch.from_numpy(q_x_a_hat).float(),
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
    ) -> torch.Tensor:  # shape: (batch_size,)
        """
        方策勾配の推定値を計算するメソッド
        Args:
            action (torch.Tensor): 選択されたアクションのテンソル (batch_size,)
            reward (torch.Tensor): 観測された報酬のテンソル (batch_size,)
            pscore (torch.Tensor): 傾向スコアのテンソル (batch_size,)
            q_x_a_hat (torch.Tensor): 期待報酬の推定値のテンソル (batch_size, n_actions)
            pi (torch.Tensor): 現在の方策による行動選択確率のテンソル (batch_size, n_actions, 1)
        Returns:
            torch.Tensor: 方策勾配の推定値のテンソル (batch_size,)
                ただし勾配計算自体はPyTorchの自動微分機能により行われるので、
                ここで返される値は 方策勾配の推定量の \nabla_{\theta} を除いた部分のみ
        """
        current_pi = pi.detach()
        log_prob = torch.log(pi + self.log_eps)
        idx_tensor = torch.arange(action.shape[0], dtype=torch.long)

        q_x_a_hat_factual = q_x_a_hat[idx_tensor, action]
        iw = current_pi[idx_tensor, action] / pscore
        estimated_policy_grad_arr = iw * (reward - q_x_a_hat_factual)
        estimated_policy_grad_arr *= log_prob[idx_tensor, action]
        estimated_policy_grad_arr += torch.sum(q_x_a_hat * current_pi * log_prob, dim=1)

        return estimated_policy_grad_arr

    def _predict_proba_as_tensor(
        self,
        context: torch.Tensor,
        action_context: torch.Tensor,
    ) -> torch.Tensor:
        """方策による行動選択確率を予測するメソッド。
        行動選択確率は各アクションのロジット値を計算し、softmax関数を適用することで得られる。
        学習時にも推論時にも利用するために、PyTorchのテンソルを入出力とする。
        Args:
            context (torch.Tensor): コンテキスト特徴量のテンソル (n_rounds, dim_context_features)
            action_context (torch.Tensor): アクション特徴量のテンソル (n_actions, dim_action_features)
        Returns:
            torch.Tensor: 行動選択確率 \pi_{\theta}(a|x) のテンソル (n_rounds, n_actions)
        """
        context_embedding = self.nn_model["context_tower"](
            context
        )  # shape: (n_rounds, dim_two_tower_embedding)
        action_embedding = self.nn_model["action_tower"](
            action_context
        )  # shape: (n_actions, dim_two_tower_embedding)

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
        context: np.ndarray,
        action_context: np.ndarray,
    ) -> np.ndarray:
        """方策による行動選択確率を予測するメソッド
        Args:
            context (np.ndarray): コンテキスト特徴量の配列 (n_rounds, dim_context_features)
            action_context (np.ndarray): アクション特徴量の配列 (n_actions, dim_action_features)
        Returns:
            np.ndarray: 行動選択確率 \pi_{\theta}(a|x) の配列 (n_rounds, n_actions, 1)
        """
        assert context.shape[1] == self.dim_context_features
        assert action_context.shape[1] == self.dim_action_features

        self.nn_model.eval()

        action_dist = self._predict_proba_as_tensor(
            context=torch.from_numpy(context).float(),
            action_context=torch.from_numpy(action_context).float(),
        )
        action_dist_ndarray = action_dist.squeeze(-1).detach().numpy()
        # open bandit pipelineの合成データクラスの仕様に合わせて、1つ軸を追加してる
        return action_dist_ndarray[:, :, np.newaxis]

    def _fit_by_regression_based_approach(
        self,
        bandit_feedback_train: BanditFeedbackDict,
        bandit_feedback_test: Optional[BanditFeedbackDict] = None,
    ) -> None:
        """two-towerモデルに基づく推薦方策を、回帰ベースアプローチで学習するメソッド。
        ここでは、報酬rの予測問題としてクロスエントロピー誤差を最小化するように学習を行う。
        Args:
            bandit_feedback_train (BanditFeedbackDict): 学習用のバンディットフィードバックデータ
            bandit_feedback_test (Optional[BanditFeedbackDict]): テスト用のバンディットフィードバックデータ
        """
        n_actions = bandit_feedback_train["n_actions"]
        context, action, reward, action_context, pscore, pi_b = (
            bandit_feedback_train["context"],
            bandit_feedback_train["action"],
            bandit_feedback_train["reward"],
            bandit_feedback_train["action_context"],
            bandit_feedback_train["pscore"],
            bandit_feedback_train["pi_b"],
        )

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
            np.zeros((reward.shape[0], n_actions)),  # 回帰ベースでは不要
        )
        action_context_tensor = torch.from_numpy(action_context).float()

        # start policy training
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
            for x, a, r, p, q_x_a_hat_ in training_data_loader:
                optimizer.zero_grad()
                # 各バッチに対するtwo-towerモデルの出力を \hat{q}(x,a) とみなす
                context_embedding = self.nn_model["context_tower"](x)
                action_embedding = self.nn_model["action_tower"](action_context_tensor)
                logits = torch.matmul(context_embedding, action_embedding.T)
                q_x_a_hat_by_two_tower = torch.sigmoid(logits)

                # 選択されたアクションに対応する\hat{q}(x,a)を取得
                selected_action_idx_tensor = torch.arange(a.shape[0], dtype=torch.long)
                q_x_a_hat_by_two_tower_of_selected_action = q_x_a_hat_by_two_tower[
                    selected_action_idx_tensor,
                    a,
                ]

                # 期待報酬の推定値 \hat{q}(x,a) と報酬rとのクロスエントロピー誤差を損失関数とする
                loss = torch.nn.functional.binary_cross_entropy(
                    q_x_a_hat_by_two_tower_of_selected_action, r
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


if __name__ == "__main__":
    # Arrange
    n_rounds = 20000
    n_actions = 4
    dim_context = 300
    dim_action_features = 200
    dim_two_tower_embedding = 120

    sut = PolicyByTwoTowerModel(
        dim_context=dim_context,
        dim_action_features=dim_action_features,
        dim_two_tower_embedding=dim_two_tower_embedding,
        is_embedding_normed=True,
        batch_size=20000,
    )

    # Act
    sut._fit_by_gradiant_based_approach(
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
