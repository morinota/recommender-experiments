from obp.dataset import OpenBanditDataset
from obp.policy import BernoulliTS
from obp.ope import OffPolicyEvaluation, InverseProbabilityWeighting as IPW

# (1) データの読み込みと前処理
dataset = OpenBanditDataset(behavior_policy="random", campaign="all")
bandit_feedback = dataset.obtain_batch_bandit_feedback()

# (2) オフライン方策シミュレーション
evaluation_policy = BernoulliTS(
    n_actions=dataset.n_actions,
    len_list=dataset.len_list,
    is_zozotown_prior=True,
    campaign="all",
    random_state=12345,
)
action_dist = evaluation_policy.compute_batch_action_dist(
    n_sim=100000, n_rounds=bandit_feedback["n_rounds"]
)

# (3) Off-Policy Evaluation
ope = OffPolicyEvaluation(bandit_feedback=bandit_feedback, ope_estimators=[IPW()])
estimated_policy_value = ope.estimate_policy_values(action_dist=action_dist)

# Randomに対するBernoulliTSの性能の改善率（相対クリック率）
relative_policy_value_of_bernoulli_ts = (
    estimated_policy_value["ipw"] / bandit_feedback["reward"].mean()
)
print(relative_policy_value_of_bernoulli_ts)
