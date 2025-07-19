from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

from recommender_experiments.service.opl.policy_strategy_interface import PolicyStrategyInterface


class ContextGNNNetwork(nn.Module):
    """
    Simple neural network for context-based recommendation.
    This is a simplified version without the full graph structure.
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
        Args:
            context: (batch_size, context_dim)
            action_context: (n_actions, action_dim)
        Returns:
            scores: (batch_size, n_actions)
        """
        batch_size = context.size(0)
        n_actions = action_context.size(0)

        # Encode context and actions
        context_encoded = self.context_encoder(context)  # (batch_size, hidden_dim)
        action_encoded = self.action_encoder(action_context)  # (n_actions, hidden_dim)

        # Expand for pairwise computation
        context_expanded = context_encoded.unsqueeze(1).expand(
            batch_size, n_actions, -1
        )  # (batch_size, n_actions, hidden_dim)
        action_expanded = action_encoded.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, n_actions, hidden_dim)

        # Concatenate and score
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
        Train the ContextGNN model using bandit feedback data.

        Args:
            bandit_feedback_train: Dictionary containing training data with keys:
                - 'context': np.ndarray of shape (n_rounds, context_dim)
                - 'action_context': np.ndarray of shape (n_actions, action_dim)
                - 'action': np.ndarray of shape (n_rounds,) - selected actions
                - 'reward': np.ndarray of shape (n_rounds,) - observed rewards
            bandit_feedback_test: Optional test data (not used in training)
        """
        # bandit_feedback_test is not used in this implementation
        del bandit_feedback_test
        context = bandit_feedback_train["context"]
        action_context = bandit_feedback_train["action_context"]
        actions = bandit_feedback_train["action"]
        rewards = bandit_feedback_train["reward"]

        # Normalize features
        context_normalized = self.context_scaler.fit_transform(context)
        action_context_normalized = self.action_scaler.fit_transform(action_context)

        # Initialize model
        context_dim = context_normalized.shape[1]
        action_dim = action_context_normalized.shape[1]
        self.model = ContextGNNNetwork(context_dim, action_dim, self.hidden_dim).to(self.device)

        # Convert to tensors
        context_tensor = torch.FloatTensor(context_normalized).to(self.device)
        action_context_tensor = torch.FloatTensor(action_context_normalized).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)

        # Training setup
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()

            # Forward pass
            scores = self.model(context_tensor, action_context_tensor)  # (n_rounds, n_actions)
            selected_scores = scores[torch.arange(len(actions_tensor)), actions_tensor]  # (n_rounds,)

            # Use rewards as binary labels (assuming rewards are 0 or 1)
            # If rewards are continuous, you might want to threshold them
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
        Predict action selection probabilities.

        Args:
            context: (n_rounds, context_dim)
            action_context: (n_actions, action_dim)
            random_state: Random seed (not used in this implementation)

        Returns:
            np.ndarray: Action probabilities of shape (n_rounds, n_actions, 1)
        """
        # random_state is not used in this implementation
        del random_state
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before making predictions")

        # Normalize features using fitted scalers
        context_normalized = self.context_scaler.transform(context)
        action_context_normalized = self.action_scaler.transform(action_context)

        # Convert to tensors
        context_tensor = torch.FloatTensor(context_normalized).to(self.device)
        action_context_tensor = torch.FloatTensor(action_context_normalized).to(self.device)

        self.model.eval()
        with torch.no_grad():
            scores = self.model(context_tensor, action_context_tensor)  # (n_rounds, n_actions)
            probs = torch.softmax(scores, dim=-1)  # Convert to probabilities

        # Return with shape (n_rounds, n_actions, 1) as expected by the interface
        return probs.cpu().numpy()[:, :, np.newaxis]

    def sample(
        self,
        context: np.ndarray,
        action_context: np.ndarray,
        random_state: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample actions according to the learned policy.

        Args:
            context: (n_rounds, context_dim)
            action_context: (n_actions, action_dim)
            random_state: Random seed

        Returns:
            tuple: (selected_actions, action_probabilities)
                - selected_actions: (n_rounds,) indices of selected actions
                - action_probabilities: (n_rounds,) probabilities of selected actions
        """
        np.random.seed(random_state)

        # Get action probabilities
        action_probs = self.predict_proba(context, action_context, random_state)  # (n_rounds, n_actions, 1)
        action_probs = action_probs[:, :, 0]  # Remove last dimension: (n_rounds, n_actions)

        n_rounds = action_probs.shape[0]
        selected_actions = np.zeros(n_rounds, dtype=int)
        selected_probs = np.zeros(n_rounds)

        # Sample actions for each round
        for i in range(n_rounds):
            probs = action_probs[i]
            selected_actions[i] = np.random.choice(len(probs), p=probs)
            selected_probs[i] = probs[selected_actions[i]]

        return selected_actions, selected_probs


if __name__ == "__main__":
    policy = PolicyByContextGnn()
    print(policy.policy_name)  # ここでは実装されていないのでエラーになる
