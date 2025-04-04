from typing import Optional, TypedDict

import numpy as np
from pydantic import BaseModel, Field, field_validator, ValidationInfo


class BanditFeedbackDict(TypedDict):
    n_rounds: int  # ラウンド数
    n_actions: int  # アクション数s
    context: np.ndarray  # 文脈 (shape: (n_rounds, dim_context))
    action_context: (
        np.ndarray
    )  # アクション特徴量 (shape: (n_actions, dim_action_features))
    action: np.ndarray  # 実際に選択されたアクション (shape: (n_rounds,))
    position: Optional[np.ndarray]  # ポジション (shape: (n_rounds,) or None)
    reward: np.ndarray  # 報酬 (shape: (n_rounds,))
    expected_reward: np.ndarray  # 期待報酬 (shape: (n_rounds, n_actions))
    pi_b: np.ndarray  # データ収集方策 P(a|x) (shape: (n_rounds, n_actions,1))
    pscore: np.ndarray  # 傾向スコア (shape: (n_rounds,))


class BanditFeedbackModel(BaseModel):
    n_rounds: int = Field(..., description="ラウンド数")
    n_actions: int = Field(..., description="アクション数")
    context: np.ndarray = Field(
        ..., description="文脈 (shape: (n_rounds, dim_context))"
    )
    action_context: np.ndarray = Field(
        ..., description="アクション特徴量 (shape: (n_actions, dim_action_features))"
    )
    action: np.ndarray = Field(
        ..., description="実際に選択されたアクション (shape: (n_rounds,))"
    )
    position: Optional[np.ndarray] = Field(
        None, description="ポジション (shape: (n_rounds,) or None)"
    )
    reward: np.ndarray = Field(..., description="報酬 (shape: (n_rounds,))")
    expected_reward: np.ndarray | None = Field(
        ..., description="期待報酬 (shape: (n_rounds, n_actions)). 現実の場合はNone"
    )
    pi_b: np.ndarray | None = Field(
        ...,
        description="データ収集方策 P(a|x) (shape: (n_rounds, n_actions, 1)). 現実の場合はNoneになり得る",
    )
    pscore: np.ndarray = Field(..., description="傾向スコア (shape: (n_rounds,))")

    class Config:
        arbitrary_types_allowed = True  # np.ndarrayを許可

    @field_validator("action", "reward", "pscore")
    def check_1d_array(cls, value: np.ndarray, field: ValidationInfo) -> np.ndarray:
        """特定のフィールドのカスタムバリデータ"""
        if value.ndim != 1:
            raise ValueError(
                f"{field.field_name} は1次元の配列を想定してます。しかし実際には{value.shape=}"
            )
        return value

    @field_validator("context", "action_context", "expected_reward")
    def check_2d_array(cls, value: np.ndarray, field: ValidationInfo) -> np.ndarray:
        if field.field_name == "expected_reward" and value is None:
            return value
        if value.ndim != 2:
            raise ValueError(
                f"{field.field_name} は2次元の配列を想定しています。しかし実際には{value.shape=}"
            )
        return value


if __name__ == "__main__":
    bandit_feedback = BanditFeedbackModel(
        n_rounds=100,
        n_actions=4,
        context=np.random.random((100, 200)),
        action_context=np.random.random((4, 150)),
        action=np.random.randint(0, 4, 100),
        position=None,
        reward=np.random.binomial(1, 0.5, 100),
        expected_reward=np.random.random((100, 4)),
        pi_b=np.random.random((100, 4)),
        pscore=np.random.random(100),
    )
    bandit_feedback_dict = bandit_feedback.model_dump()
