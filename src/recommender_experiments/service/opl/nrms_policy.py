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
class NRMSPolicyLearner(NNPolicyLearner):
    dim_context: int
    dim_action_features: int
    hidden_layer_size: tuple[int, ...] = (100,)
    activation: str = "relu"
    dim_two_tower_embedding: int = 256
    max_iter: int = 200
    learning_rate_init: float = 0.0001
    batch_size: int = 32
    n_heads: int = 16
    additive_attn_hidden_dim: int = 200

    def __post_init__(self):
        print("Initializing NRMSPolicyLearner...")

        activation_layer = self._get_activation_layer()

        # Context Tower (User Encoder)
        self.user_encoder = UserEncoder(
            hidden_size=self.dim_two_tower_embedding,
            multihead_attn_num_heads=self.n_heads,
            additive_attn_hidden_dim=self.additive_attn_hidden_dim,
        )

        # Action Tower (News Encoder)
        self.news_encoder = PLMBasedNewsEncoder(
            pretrained="bert-base-uncased",
            multihead_attn_num_heads=self.n_heads,
            additive_attn_hidden_dim=self.additive_attn_hidden_dim,
        )

        # Combine all layers
        self.nn_model = nn.ModuleDict(
            {
                "user_encoder": self.user_encoder,
                "news_encoder": self.news_encoder,
            }
        )

    def _predict_proba_for_fit(
        self, context: torch.Tensor, action_context: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict action probabilities based on user context and candidate news.
        """
        # Encode user context
        user_embedding = self.nn_model["user_encoder"](context, self.news_encoder)

        # Encode action candidates
        action_embedding = self.nn_model["news_encoder"](action_context)

        # 各contextについて、各actionのスコアを計算
        scores = torch.matmul(user_embedding, action_embedding.T)

        # softmax関数を適用して、行動選択確率分布を得る
        pi = torch.softmax(scores, dim=1)

        return pi.unsqueeze(-1)

    def fit(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        action_context: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
    ) -> None:
        print("Starting training...")

        # Set default propensity scores if not provided
        if pscore is None:
            pscore = np.ones_like(action) / action_context.shape[0]
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
        for _ in tqdm(range(self.max_iter), desc="policy learning"):
            self.nn_model.train()
            for x, a, r, p, pos in training_data_loader:
                optimizer.zero_grad()
                # 新方策の行動選択確率分布\pi(a|x)を計算
                pi = self._predict_proba_for_fit(x, action_context_tensor)

                # IPS推定量を用いて、方策勾配の推定値を計算
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

                # lossを最小化するようにモデルパラメータを更新
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


class PLMBasedNewsEncoder(nn.Module):
    def __init__(
        self,
        pretrained: str = "bert-base-uncased",
        multihead_attn_num_heads: int = 16,
        additive_attn_hidden_dim: int = 200,
    ):
        super().__init__()
        self.plm = AutoModel.from_pretrained(pretrained)

        plm_hidden_size = AutoConfig.from_pretrained(pretrained).hidden_size

        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=plm_hidden_size,
            num_heads=multihead_attn_num_heads,
            batch_first=True,
        )
        self.additive_attention = AdditiveAttention(
            plm_hidden_size, additive_attn_hidden_dim
        )

    def forward(self, input_val: torch.Tensor) -> torch.Tensor:
        V = self.plm(
            input_val
        ).last_hidden_state  # [batch_size, seq_len] -> [batch_size, seq_len, hidden_size]
        multihead_attn_output, _ = self.multihead_attention(
            V, V, V
        )  # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, hidden_size]
        additive_attn_output = self.additive_attention(
            multihead_attn_output
        )  # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, hidden_size]
        output = torch.sum(
            additive_attn_output, dim=1
        )  # [batch_size, seq_len, hidden_size] -> [batch_size, hidden_size]

        return output


import torch
from torch import nn


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class AdditiveAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(
                input_dim, hidden_dim
            ),  # in: (batch_size, seq_len, input_dim), out: (batch_size, seq_len, hidden_dim)
            nn.Tanh(),  # in: (batch_size, seq_len, hidden_dim), out: (batch_size, seq_len, hidden_dim)
            nn.Linear(
                hidden_dim, 1, bias=False
            ),  # in: (batch_size, seq_len, hidden_dim), out: (batch_size, seq_len, 1)
            nn.Softmax(dim=-2),
        )
        self.attention.apply(init_weights)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        attention_weight = self.attention(input)
        return input * attention_weight


class UserEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        multihead_attn_num_heads: int = 16,
        additive_attn_hidden_dim: int = 200,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=multihead_attn_num_heads, batch_first=True
        )
        self.additive_attention = AdditiveAttention(
            hidden_size, additive_attn_hidden_dim
        )

    def forward(
        self, news_histories: torch.Tensor, news_encoder: nn.Module
    ) -> torch.Tensor:
        batch_size, hist_size, seq_len = news_histories.size()
        news_histories = news_histories.view(
            batch_size * hist_size, seq_len
        )  # [batch_size, hist_size, seq_len] -> [batch_size*hist_size, seq_len]

        news_histories_encoded = news_encoder(
            news_histories
        )  # [batch_size*hist_size, seq_len] -> [batch_size*hist_size, emb_dim]

        news_histories_encoded = news_histories_encoded.view(
            batch_size, hist_size, self.hidden_size
        )  # [batch_size*hist_size, seq_len] -> [batch_size, hist_size, emb_dim]

        multihead_attn_output, _ = self.multihead_attention(
            news_histories_encoded, news_histories_encoded, news_histories_encoded
        )  # [batch_size, hist_size, emb_dim] -> [batch_size, hist_size, emb_dim]

        additive_attn_output = self.additive_attention(
            multihead_attn_output
        )  # [batch_size, hist_size, emb_dim] -> [batch_size, emb_dim]

        output = torch.sum(additive_attn_output, dim=1)

        return output


class NRMS(nn.Module):
    def __init__(
        self,
        news_encoder: nn.Module,
        user_encoder: nn.Module,
        hidden_size: int,
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
    ) -> None:
        super().__init__()
        self.news_encoder: nn.Module = news_encoder
        self.user_encoder: nn.Module = user_encoder
        self.hidden_size: int = hidden_size
        self.loss_fn = loss_fn

    def forward(
        self,
        candidate_news: torch.Tensor,
        news_histories: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        candidate_news : torch.Tensor (shape = (batch_size, candidate_num, seq_len))
        news_histories : torch.Tensor (shape = (batch_size, candidate_num, seq_len))
        ===========================================================================

        Returns
        ----------
        output: torch.Tensor (shape = (batch_size, candidate_num))
        """

        batch_size, candidate_num, seq_len = candidate_news.size()
        candidate_news = candidate_news.view(batch_size * candidate_num, seq_len)
        news_candidate_encoded = self.news_encoder(
            candidate_news
        )  # [batch_size * (candidate_num), seq_len] -> [batch_size * (candidate_num), emb_dim]
        news_candidate_encoded = news_candidate_encoded.view(
            batch_size, candidate_num, self.hidden_size
        )  # [batch_size * (candidate_num), emb_dim] -> [batch_size, (candidate_num), emb_dim]

        news_histories_encoded = self.user_encoder(
            news_histories, self.news_encoder
        )  # [batch_size, histories, seq_len] -> [batch_size, emb_dim]
        news_histories_encoded = news_histories_encoded.unsqueeze(
            -1
        )  # [batch_size, emb_dim] -> [batch_size, emb_dim, 1]

        output = torch.bmm(
            news_candidate_encoded, news_histories_encoded
        )  # [batch_size, (candidate_num), emb_dim] x [batch_size, emb_dim, 1] -> [batch_size, (1+npratio), 1, 1]
        output = output.squeeze(-1).squeeze(
            -1
        )  # [batch_size, (1+npratio), 1, 1] -> [batch_size, (1+npratio)]

        # NOTE:
        # when "val" mode(self.training == False) → not calculate loss score
        # Multiple hot labels may exist on target.
        # e.g.
        # candidate_news = ["N24510","N39237","N9721"]
        # target = [0,2](=[1, 0, 1] in one-hot format)
        if not self.training:
            return ModelOutput(logits=output, loss=torch.Tensor([-1]), labels=target)

        loss = self.loss_fn(output, target)
        return ModelOutput(logits=output, loss=loss, labels=target)
