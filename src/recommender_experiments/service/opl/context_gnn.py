from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

from recommender_experiments.service.opl.policy_strategy_interface import PolicyStrategyInterface


class ContextGNNNetwork(nn.Module):
    """
    コンテキストベースの推薦のためのシンプルなニューラルネットワーク。
    完全なグラフ構造を持たない簡略化されたバージョン。
    """

    def __init__(self, context_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )

        self.scorer = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, context: torch.Tensor, action_context: torch.Tensor) -> torch.Tensor:
        """
        フォワードパス処理
        
        Args:
            context: ユーザーコンテキスト特徴量 (batch_size, context_dim)
            action_context: アクション特徴量 (n_actions, action_dim)
        Returns:
            scores: 各アクションに対するスコア (batch_size, n_actions)
        """
        batch_size = context.size(0)
        n_actions = action_context.size(0)

        # コンテキストとアクションをエンコード
        context_encoded = self.context_encoder(context)  # (batch_size, hidden_dim)
        action_encoded = self.action_encoder(action_context)  # (n_actions, hidden_dim)

        # ペアワイズ計算のために次元を拡張
        context_expanded = context_encoded.unsqueeze(1).expand(
            batch_size, n_actions, -1
        )  # (batch_size, n_actions, hidden_dim)
        action_expanded = action_encoded.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, n_actions, hidden_dim)

        # 結合してスコアを計算
        combined = torch.cat([context_expanded, action_expanded], dim=-1)  # (batch_size, n_actions, hidden_dim * 2)
        scores = self.scorer(combined).squeeze(-1)  # (batch_size, n_actions)

        return scores


class PolicyByContextGnn(PolicyStrategyInterface):
    def __init__(self, hidden_dim: int = 64, lr: float = 0.001, epochs: int = 100, device: str = "cpu") -> None:
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device(device)
        self.model: Optional[ContextGNNNetwork] = None
        self.context_scaler = StandardScaler()
        self.action_scaler = StandardScaler()
        self.is_fitted = False

    @property
    def policy_name(self) -> str:
        return "ContextGNNPolicy"

    def fit(
        self,
        bandit_feedback_train: dict,
        bandit_feedback_test: dict | None = None,
    ) -> None:
        """
        バンディットフィードバックデータを使ってContextGNNモデルを学習する。

        Args:
            bandit_feedback_train: 学習データを含む辞書（以下のキーを含む）:
                - 'context': ユーザーコンテキスト特徴量 (n_rounds, context_dim)
                - 'action_context': アクション特徴量 (n_actions, action_dim)
                - 'action': 選択されたアクション (n_rounds,)
                - 'reward': 観測された報酬 (n_rounds,)
            bandit_feedback_test: テストデータ（この実装では使用しない）
        """
        # bandit_feedback_testはこの実装では使用しない
        del bandit_feedback_test
        context = bandit_feedback_train["context"]
        action_context = bandit_feedback_train["action_context"]
        actions = bandit_feedback_train["action"]
        rewards = bandit_feedback_train["reward"]

        # 特徴量の正規化
        context_normalized = self.context_scaler.fit_transform(context)
        action_context_normalized = self.action_scaler.fit_transform(action_context)

        # モデルの初期化
        context_dim = context_normalized.shape[1]
        action_dim = action_context_normalized.shape[1]
        self.model = ContextGNNNetwork(context_dim, action_dim, self.hidden_dim).to(self.device)

        # テンソルに変換
        context_tensor = torch.FloatTensor(context_normalized).to(self.device)
        action_context_tensor = torch.FloatTensor(action_context_normalized).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)

        # 学習の設定
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        # 学習ループ
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()

            # フォワードパス
            scores = self.model(context_tensor, action_context_tensor)  # (n_rounds, n_actions)
            selected_scores = scores[torch.arange(len(actions_tensor)), actions_tensor]  # (n_rounds,)

            # 報酬をバイナリラベルとして使用（報酬が0または1と仮定）
            # 連続値の報酬の場合は閾値処理を検討
            binary_rewards = (rewards_tensor > 0).float()

            loss = criterion(selected_scores, binary_rewards)
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        self.is_fitted = True

    def predict_proba(
        self,
        context: np.ndarray,
        action_context: np.ndarray,
        random_state: int = 0,
    ) -> np.ndarray:
        """
        アクション選択確率を予測する。

        Args:
            context: ユーザーコンテキスト特徴量 (n_rounds, context_dim)
            action_context: アクション特徴量 (n_actions, action_dim)
            random_state: 乱数シード（この実装では使用しない）

        Returns:
            np.ndarray: アクション選択確率 (n_rounds, n_actions, 1)
        """
        # random_stateはこの実装では使用しない
        del random_state
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before making predictions")

        # 学習済みスケーラーを使って特徴量を正規化
        context_normalized = self.context_scaler.transform(context)
        action_context_normalized = self.action_scaler.transform(action_context)

        # テンソルに変換
        context_tensor = torch.FloatTensor(context_normalized).to(self.device)
        action_context_tensor = torch.FloatTensor(action_context_normalized).to(self.device)

        self.model.eval()
        with torch.no_grad():
            scores = self.model(context_tensor, action_context_tensor)  # (n_rounds, n_actions)
            probs = torch.softmax(scores, dim=-1)  # 確率に変換

        # インターフェースで期待される形状 (n_rounds, n_actions, 1) で返す
        return probs.cpu().numpy()[:, :, np.newaxis]

    def sample(
        self,
        context: np.ndarray,
        action_context: np.ndarray,
        random_state: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        学習した方策に従ってアクションをサンプリングする。

        Args:
            context: ユーザーコンテキスト特徴量 (n_rounds, context_dim)
            action_context: アクション特徴量 (n_actions, action_dim)
            random_state: 乱数シード

        Returns:
            tuple: (選択アクション, アクション確率)
                - selected_actions: 選択されたアクションのインデックス (n_rounds,)
                - action_probabilities: 選択されたアクションの確率 (n_rounds,)
        """
        np.random.seed(random_state)

        # アクション選択確率を取得
        action_probs = self.predict_proba(context, action_context, random_state)  # (n_rounds, n_actions, 1)
        action_probs = action_probs[:, :, 0]  # 最後の次元を削除: (n_rounds, n_actions)

        n_rounds = action_probs.shape[0]
        selected_actions = np.zeros(n_rounds, dtype=int)
        selected_probs = np.zeros(n_rounds)

        # 各ラウンドでアクションをサンプリング
        for i in range(n_rounds):
            probs = action_probs[i]
            selected_actions[i] = np.random.choice(len(probs), p=probs)
            selected_probs[i] = probs[selected_actions[i]]

        return selected_actions, selected_probs


if __name__ == "__main__":
    policy = PolicyByContextGnn()
    print(policy.policy_name)  # ここでは実装されていないのでエラーになる
