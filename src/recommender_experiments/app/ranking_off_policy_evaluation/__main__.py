import random
from typing import Callable, Optional, TypedDict
from obp.dataset import OpenBanditDataset, SyntheticBanditDataset
from obp.policy import BernoulliTS, LogisticTS, LogisticUCB
from obp.ope import (
    ReplayMethod,
    InverseProbabilityWeighting,
    BaseOffPolicyEstimator,
    OffPolicyEvaluation,
    InverseProbabilityWeighting as IPW,
)

import polars as pl

# 実際にzozotownで収集されたバンディットフィードバックデータの使い方確認

# データセットクラスのインスタンス化
dataset = OpenBanditDataset(behavior_policy="random", campaign="all")
# 属性の確認
print(f"Number of Rounds: {dataset.n_rounds}")
print(f"Number of Actions: {dataset.n_actions}")
print(f"slate size: {dataset.len_list}")

# バンディットフィードバックを辞書形式で取得
bandit_feedback, _ = dataset.obtain_batch_bandit_feedback(
    test_size=0.3, is_timeseries_split=True
)
print(bandit_feedback.keys())

# データ収集方策の性能のon-policy評価結果
print(f"{OpenBanditDataset.calc_on_policy_policy_value_estimate('bts', 'all')}")
print(f"{OpenBanditDataset.calc_on_policy_policy_value_estimate('random', 'all')}")
print("--------------------")

# バンディットフィードバックの中身を確認
for round_idx in range(3):
    print(f"Round: {round_idx}")
    print(f"Context: {bandit_feedback['context'][round_idx]}")
    print(f"Position: {bandit_feedback['position'][round_idx]}")
    print(f"Action: {bandit_feedback['action'][round_idx]}")
    print(f"Action context: {bandit_feedback['action_context'][round_idx]}")
    print(f"Reward: {bandit_feedback['reward'][round_idx]}")
    print(f"Propensity Score: {bandit_feedback['pscore'][round_idx]}")
    print("--------------------")

# context-freeなbanditモデルのオフ方策学習 & オフ方策評価
evaluation_policy = BernoulliTS(
    n_actions=dataset.n_actions,
    len_list=dataset.len_list,
    is_zozotown_prior=False,
    # zozoが用意した事前分布のパラメータを使う場合
    # is_zozotown_prior=True,
    # campaign="all",
    # random_state=12345,
)
## パラメータ更新
for round_idx in range(bandit_feedback["n_rounds"]):
    action = bandit_feedback["action"][round_idx]
    reward = bandit_feedback["reward"][round_idx]
    evaluation_policy.update_params(action=action, reward=reward)
action_dist = evaluation_policy.compute_batch_action_dist(
    n_rounds=bandit_feedback["n_rounds"],
    n_sim=100000,
)
## オフ方策評価
ope = OffPolicyEvaluation(bandit_feedback=bandit_feedback, ope_estimators=[IPW()])
estimated_policy_value = ope.estimate_policy_values(action_dist=action_dist)
## オフ方策学習した新方策の性能のオフライン評価値と、データ収集方策の性能のオンライン評価値を比較
relative_policy_value_of_bernoulli_ts = (
    estimated_policy_value["ipw"] / bandit_feedback["reward"].mean()
)
print(f"{relative_policy_value_of_bernoulli_ts=}")

# contexual banditモデルのオフ方策学習 & オフ方策評価
new_policy = LogisticTS(
    n_actions=dataset.n_actions,
    dim=dataset.dim_context,
    len_list=dataset.len_list,
)
# パラメータ更新
for round_idx in range(bandit_feedback["n_rounds"]):
    context = bandit_feedback["context"][round_idx]
    action = bandit_feedback["action"][round_idx]
    reward = bandit_feedback["reward"][round_idx]
    new_policy.update_params(
        action=action,
        reward=reward,
        # contextベクトルの形状を、明示的に(dim_context, )から(1, dim_context)に変換しておく
        context=context.reshape(1, -1),
    )

action_dist = new_policy.select_action(
    # contextベクトルの形状を(1, dim_context)に変換
    context=bandit_feedback["context"][0].reshape(1, -1),
)
print(f"{action_dist=}")  # shape=(len_list, )の配列。各要素は選択されたアクションのid
print(f"{action_dist.shape=}")
