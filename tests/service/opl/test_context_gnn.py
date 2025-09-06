import numpy as np
import pytest
import torch

from recommender_experiments.service.opl.context_gnn import (
    ContextGNNNetwork,
    PolicyByContextGnn,
)
from recommender_experiments.service.synthetic_bandit_feedback import (
    BanditFeedbackModel,
)


class TestContextGNNNetwork:
    def test_初期化時に正しい構造が作成されること(self):
        # Arrange & Act
        context_dim = 10
        action_dim = 5
        hidden_dim = 32
        model = ContextGNNNetwork(context_dim, action_dim, hidden_dim)

        # Assert
        assert isinstance(model.context_encoder, torch.nn.Sequential)
        assert isinstance(model.action_encoder, torch.nn.Sequential)
        assert isinstance(model.scorer, torch.nn.Sequential)

    def test_forward時に正しい形状の出力が返されること(self):
        # Arrange
        context_dim = 10
        action_dim = 5
        batch_size = 3
        n_actions = 4
        model = ContextGNNNetwork(context_dim, action_dim)

        context = torch.randn(batch_size, context_dim)
        action_context = torch.randn(n_actions, action_dim)

        # Act
        scores = model(context, action_context)

        # Assert
        assert scores.shape == (batch_size, n_actions), "出力の形状が(batch_size, n_actions)であること"
        assert isinstance(scores, torch.Tensor), "出力がTensorであること"


class TestPolicyByContextGnn:
    def test_初期化時に正しいプロパティが設定されること(self):
        # Arrange & Act
        policy = PolicyByContextGnn(hidden_dim=64, lr=0.01, epochs=50, device="cpu")

        # Assert
        assert policy.hidden_dim == 64
        assert policy.lr == 0.01
        assert policy.epochs == 50
        assert policy.device.type == "cpu"
        assert policy.policy_name == "ContextGNNPolicy"
        assert policy.is_fitted is False

    def test_fit後にモデルが学習状態になること(self):
        # Arrange
        n_actions = 4
        dim_context = 10
        dim_action = 8
        policy = PolicyByContextGnn(hidden_dim=32, epochs=5)

        bandit_feedback_train = {
            "context": np.random.random((100, dim_context)),
            "action_context": np.random.random((n_actions, dim_action)),
            "action": np.random.randint(0, n_actions, 100),
            "reward": np.random.binomial(1, 0.5, 100),
        }

        # Act
        policy.fit(bandit_feedback_train)

        # Assert
        assert policy.is_fitted is True, "学習完了フラグが立つこと"
        assert policy.model is not None, "モデルが初期化されること"
        assert isinstance(policy.model, ContextGNNNetwork), "正しいモデル型であること"

    def test_predict_proba時に正しい形状の確率分布が返されること(self):
        # Arrange
        n_actions = 4
        dim_context = 10
        dim_action = 8
        policy = PolicyByContextGnn(hidden_dim=32, epochs=5)

        # 学習データで事前学習
        bandit_feedback_train = {
            "context": np.random.random((100, dim_context)),
            "action_context": np.random.random((n_actions, dim_action)),
            "action": np.random.randint(0, n_actions, 100),
            "reward": np.random.binomial(1, 0.5, 100),
        }
        policy.fit(bandit_feedback_train)

        # テスト用データ
        test_context = np.random.random((5, dim_context))
        test_action_context = np.random.random((n_actions, dim_action))

        # Act
        action_probs = policy.predict_proba(test_context, test_action_context)

        # Assert
        assert isinstance(action_probs, np.ndarray), "確率分布はndarrayであること"
        assert action_probs.shape == (5, n_actions, 1), "確率分布の形状が(n_rounds, n_actions, 1)であること"
        assert np.all(action_probs >= 0), "確率は非負であること"
        assert np.all(action_probs <= 1), "確率は1以下であること"
        # 各ラウンドで確率の和が1になること
        prob_sums = action_probs[:, :, 0].sum(axis=1)
        np.testing.assert_allclose(prob_sums, 1.0, rtol=1e-5)

    def test_sample時に正しい形状の行動選択が返されること(self):
        # Arrange
        n_actions = 4
        dim_context = 10
        dim_action = 8
        policy = PolicyByContextGnn(hidden_dim=32, epochs=5)

        # 学習データで事前学習
        bandit_feedback_train = {
            "context": np.random.random((100, dim_context)),
            "action_context": np.random.random((n_actions, dim_action)),
            "action": np.random.randint(0, n_actions, 100),
            "reward": np.random.binomial(1, 0.5, 100),
        }
        policy.fit(bandit_feedback_train)

        # テスト用データ
        test_context = np.random.random((5, dim_context))
        test_action_context = np.random.random((n_actions, dim_action))

        # Act
        selected_actions, selected_probs = policy.sample(test_context, test_action_context, random_state=42)

        # Assert
        assert isinstance(selected_actions, np.ndarray), "選択行動はndarrayであること"
        assert isinstance(selected_probs, np.ndarray), "選択確率はndarrayであること"
        assert selected_actions.shape == (5,), "選択行動の形状が(n_rounds,)であること"
        assert selected_probs.shape == (5,), "選択確率の形状が(n_rounds,)であること"
        assert np.all(selected_actions >= 0), "選択行動は非負であること"
        assert np.all(selected_actions < n_actions), "選択行動は有効な範囲内であること"
        assert np.all(selected_probs >= 0), "選択確率は非負であること"
        assert np.all(selected_probs <= 1), "選択確率は1以下であること"

    def test_学習前のpredict_probaでエラーが発生すること(self):
        # Arrange
        policy = PolicyByContextGnn()
        test_context = np.random.random((5, 10))
        test_action_context = np.random.random((4, 8))

        # Act & Assert
        with pytest.raises(ValueError, match="Model must be fitted before making predictions"):
            policy.predict_proba(test_context, test_action_context)

    def test_学習前のsampleでエラーが発生すること(self):
        # Arrange
        policy = PolicyByContextGnn()
        test_context = np.random.random((5, 10))
        test_action_context = np.random.random((4, 8))

        # Act & Assert
        with pytest.raises(ValueError, match="Model must be fitted before making predictions"):
            policy.sample(test_context, test_action_context)

    def test_同じrandom_stateで同じ結果が得られること(self):
        # Arrange
        n_actions = 4
        dim_context = 10
        dim_action = 8
        policy = PolicyByContextGnn(hidden_dim=32, epochs=5)

        # 学習データで事前学習
        bandit_feedback_train = {
            "context": np.random.random((100, dim_context)),
            "action_context": np.random.random((n_actions, dim_action)),
            "action": np.random.randint(0, n_actions, 100),
            "reward": np.random.binomial(1, 0.5, 100),
        }
        policy.fit(bandit_feedback_train)

        test_context = np.random.random((5, dim_context))
        test_action_context = np.random.random((n_actions, dim_action))

        # Act
        actions1, probs1 = policy.sample(test_context, test_action_context, random_state=42)
        actions2, probs2 = policy.sample(test_context, test_action_context, random_state=42)

        # Assert
        np.testing.assert_array_equal(actions1, actions2, "同じrandom_stateで同じ行動が選択されること")
        np.testing.assert_array_equal(probs1, probs2, "同じrandom_stateで同じ確率が得られること")

    def test_改良版パラメータで初期化できること(self):
        # Arrange & Act
        policy = PolicyByContextGnn(
            hidden_dim=128,
            dropout_rate=0.2,
            num_layers=4,
            batch_size=16,
            early_stopping_patience=5,
            use_continuous_rewards=True,
        )

        # Assert
        assert policy.hidden_dim == 128
        assert policy.dropout_rate == 0.2
        assert policy.num_layers == 4
        assert policy.batch_size == 16
        assert policy.early_stopping_patience == 5
        assert policy.use_continuous_rewards is True

    def test_連続値報酬で学習できること(self):
        # Arrange
        n_actions = 4
        dim_context = 10
        dim_action = 8
        policy = PolicyByContextGnn(
            hidden_dim=32, epochs=3, batch_size=16, use_continuous_rewards=True, early_stopping_patience=2
        )

        bandit_feedback_train = {
            "context": np.random.random((100, dim_context)),
            "action_context": np.random.random((n_actions, dim_action)),
            "action": np.random.randint(0, n_actions, 100),
            "reward": np.random.uniform(0.0, 1.0, 100),  # 連続値報酬
        }

        # Act
        policy.fit(bandit_feedback_train)

        # Assert
        assert policy.is_fitted is True
        assert policy.use_continuous_rewards is True

    def test_学習履歴を取得できること(self):
        # Arrange
        n_actions = 4
        dim_context = 10
        dim_action = 8
        policy = PolicyByContextGnn(hidden_dim=32, epochs=3, batch_size=16)

        bandit_feedback_train = {
            "context": np.random.random((100, dim_context)),
            "action_context": np.random.random((n_actions, dim_action)),
            "action": np.random.randint(0, n_actions, 100),
            "reward": np.random.binomial(1, 0.5, 100),
        }

        # Act
        policy.fit(bandit_feedback_train)
        history = policy.get_training_history()

        # Assert
        assert isinstance(history, dict)
        assert "loss" in history
        assert "val_loss" in history
        assert len(history["loss"]) > 0
        assert len(history["val_loss"]) > 0
