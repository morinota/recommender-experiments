from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

from recommender_experiments.service.opl.policy_strategy_interface import PolicyStrategyInterface


class ContextGNNNetwork(nn.Module):
    """
    コンテキストベースの推薦のための改良されたニューラルネットワーク。
    Attention機構、Dropout、BatchNormを含む高度なアーキテクチャ。
    """

    def __init__(
        self, context_dim: int, action_dim: int, hidden_dim: int = 64, dropout_rate: float = 0.1, num_layers: int = 3
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # コンテキストエンコーダー（改良版）
        context_layers = []
        context_layers.append(nn.Linear(context_dim, hidden_dim))
        context_layers.append(nn.BatchNorm1d(hidden_dim))
        context_layers.append(nn.ReLU())
        context_layers.append(nn.Dropout(dropout_rate))

        for _ in range(num_layers - 1):
            context_layers.extend(
                [nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate)]
            )
        self.context_encoder = nn.Sequential(*context_layers)

        # アクションエンコーダー（改良版）
        action_layers = []
        action_layers.append(nn.Linear(action_dim, hidden_dim))
        action_layers.append(nn.BatchNorm1d(hidden_dim))
        action_layers.append(nn.ReLU())
        action_layers.append(nn.Dropout(dropout_rate))

        for _ in range(num_layers - 1):
            action_layers.extend(
                [nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate)]
            )
        self.action_encoder = nn.Sequential(*action_layers)

        # Attention機構
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, dropout=dropout_rate, batch_first=True
        )

        # クロス特徴量生成
        self.cross_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),  # concat + cross product
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # 最終スコア計算
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, context: torch.Tensor, action_context: torch.Tensor) -> torch.Tensor:
        """
        改良されたフォワードパス処理

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

        # Attention機構でコンテキスト-アクション間の関係性を学習
        # アクションをkey/value、コンテキストをqueryとして使用
        action_for_attention = action_encoded.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # (batch_size, n_actions, hidden_dim)
        context_for_attention = context_encoded.unsqueeze(1)  # (batch_size, 1, hidden_dim)

        attended_context, _ = self.attention(
            context_for_attention, action_for_attention, action_for_attention
        )  # (batch_size, 1, hidden_dim), (batch_size, 1, n_actions)
        attended_context = attended_context.squeeze(1)  # (batch_size, hidden_dim)

        # ペアワイズ計算のために次元を拡張
        context_expanded = context_encoded.unsqueeze(1).expand(batch_size, n_actions, -1)
        action_expanded = action_encoded.unsqueeze(0).expand(batch_size, -1, -1)

        # クロス特徴量：要素積を計算
        cross_features = context_expanded * action_expanded  # (batch_size, n_actions, hidden_dim)

        # 特徴量を結合：concat + cross + attended
        combined = torch.cat(
            [
                context_expanded,  # 元のコンテキスト
                action_expanded,  # 元のアクション
                cross_features,  # クロス特徴量
            ],
            dim=-1,
        )  # (batch_size, n_actions, hidden_dim * 3)

        # バッチ次元を統合してクロス特徴量ネットワークに通す
        combined_flat = combined.view(-1, self.hidden_dim * 3)  # (batch_size * n_actions, hidden_dim * 3)
        cross_output = self.cross_net(combined_flat)  # (batch_size * n_actions, hidden_dim)
        cross_output = cross_output.view(batch_size, n_actions, -1)  # (batch_size, n_actions, hidden_dim)

        # 最終スコア計算
        scores_flat = self.scorer(cross_output.view(-1, self.hidden_dim)).squeeze(-1)  # (batch_size * n_actions,)
        scores = scores_flat.view(batch_size, n_actions)  # (batch_size, n_actions)

        return scores


class PolicyByContextGnn(PolicyStrategyInterface):
    def __init__(
        self,
        hidden_dim: int = 64,
        lr: float = 0.001,
        epochs: int = 100,
        device: str = "cpu",
        dropout_rate: float = 0.1,
        num_layers: int = 3,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
        use_continuous_rewards: bool = True,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device(device)
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.use_continuous_rewards = use_continuous_rewards

        self.model: Optional[ContextGNNNetwork] = None
        self.context_scaler = StandardScaler()
        self.action_scaler = StandardScaler()
        self.is_fitted = False
        self.training_history = {"loss": [], "val_loss": []}

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

        # 訓練/検証データの分割（8:2）
        n_samples = len(context_normalized)
        train_size = int(0.8 * n_samples)
        indices = np.random.permutation(n_samples)
        train_idx, val_idx = indices[:train_size], indices[train_size:]

        # モデルの初期化
        context_dim = context_normalized.shape[1]
        action_dim = action_context_normalized.shape[1]
        self.model = ContextGNNNetwork(context_dim, action_dim, self.hidden_dim, self.dropout_rate, self.num_layers).to(
            self.device
        )

        # 学習の設定
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        # 損失関数の選択
        if self.use_continuous_rewards:
            criterion = nn.MSELoss()  # 連続値報酬用
        else:
            criterion = nn.BCEWithLogitsLoss()  # バイナリ報酬用

        # Early Stopping用の変数
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        # 学習ループ（ミニバッチ対応）
        for epoch in range(self.epochs):
            # 訓練フェーズ
            self.model.train()
            train_loss = 0.0
            num_train_batches = 0

            for i in range(0, len(train_idx), self.batch_size):
                batch_idx = train_idx[i : i + self.batch_size]

                # バッチデータの準備
                context_batch = torch.FloatTensor(context_normalized[batch_idx]).to(self.device)
                action_context_tensor = torch.FloatTensor(action_context_normalized).to(self.device)
                actions_batch = torch.LongTensor(actions[batch_idx]).to(self.device)
                rewards_batch = torch.FloatTensor(rewards[batch_idx]).to(self.device)

                optimizer.zero_grad()

                # フォワードパス
                scores = self.model(context_batch, action_context_tensor)
                selected_scores = scores[torch.arange(len(actions_batch)), actions_batch]

                # 損失計算
                if self.use_continuous_rewards:
                    # 連続値報酬：MSE Loss
                    loss = criterion(selected_scores, rewards_batch)
                else:
                    # バイナリ報酬：BCE Loss
                    binary_rewards = (rewards_batch > 0).float()
                    loss = criterion(selected_scores, binary_rewards)

                loss.backward()

                # 勾配クリッピング
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += loss.item()
                num_train_batches += 1

            avg_train_loss = train_loss / num_train_batches

            # 検証フェーズ
            self.model.eval()
            val_loss = 0.0
            num_val_batches = 0

            with torch.no_grad():
                for i in range(0, len(val_idx), self.batch_size):
                    batch_idx = val_idx[i : i + self.batch_size]

                    context_batch = torch.FloatTensor(context_normalized[batch_idx]).to(self.device)
                    action_context_tensor = torch.FloatTensor(action_context_normalized).to(self.device)
                    actions_batch = torch.LongTensor(actions[batch_idx]).to(self.device)
                    rewards_batch = torch.FloatTensor(rewards[batch_idx]).to(self.device)

                    scores = self.model(context_batch, action_context_tensor)
                    selected_scores = scores[torch.arange(len(actions_batch)), actions_batch]

                    if self.use_continuous_rewards:
                        loss = criterion(selected_scores, rewards_batch)
                    else:
                        binary_rewards = (rewards_batch > 0).float()
                        loss = criterion(selected_scores, binary_rewards)

                    val_loss += loss.item()
                    num_val_batches += 1

            avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else float("inf")

            # 学習率調整
            scheduler.step(avg_val_loss)

            # 履歴の記録
            self.training_history["loss"].append(avg_train_loss)
            self.training_history["val_loss"].append(avg_val_loss)

            # Early Stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # Early Stoppingの判定
            if patience_counter >= self.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # 最良のモデルを復元
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        self.is_fitted = True
        print(f"学習完了。最終検証損失: {best_val_loss:.4f}")

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

    def get_training_history(self) -> dict:
        """
        学習履歴を取得する。

        Returns:
            dict: 学習履歴（loss, val_lossのリスト）
        """
        return self.training_history.copy()


if __name__ == "__main__":
    policy = PolicyByContextGnn()
    print(policy.policy_name)  # ここでは実装されていないのでエラーになる
