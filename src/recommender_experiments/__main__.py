from obp.dataset import OpenBanditDataset
from obp.ope import InverseProbabilityWeighting as IPW
from obp.ope import OffPolicyEvaluation
from obp.policy import BernoulliTS

# (1) データの読み込みと前処理
## 「全アイテムキャンペーン」においてRandom policyが集めたログデータを読み込む(これらは引数に設定)
dataset = OpenBanditDataset(behavior_policy="random", campaign="all")
## 過去の意思決定policyによる蓄積データ`bandit feedback`を得る
bandit_feedback = dataset.obtain_batch_bandit_feedback()
print(bandit_feedback.keys())

# (2) オフライン方策シミュレーション
## 評価方策として、BernoulliTSを用いる
evaluation_policy = BernoulliTS(
    n_actions=dataset.n_actions,
    len_list=dataset.len_list,
    is_zozotown_prior=True,  # ZOZOTOWN上での挙動を再現
    campaign="all",
    random_state=12345,
)
## シミュレーションにより、BernoulliTSによる行動選択確率を算出
action_dist = evaluation_policy.compute_batch_action_dist(n_sim=100000, n_rounds=bandit_feedback["n_rounds"])

# (3) Off-Policy Evaluation
## 算出された評価方策の行動選択確率を用いて、IPW推定量を用いてオフライン評価を行う
## OffPolicyEvaluationクラスの初期化時には、過去の意思決定policyによる蓄積データと、OPE推定量(複数設定可能)を渡す
ope = OffPolicyEvaluation(bandit_feedback=bandit_feedback, ope_estimators=[IPW()])
## 設定されたOPE推定量ごとの推定値を含んだ、辞書が返される
estimated_policy_value = ope.estimate_policy_values(action_dist=action_dist)

# Randomに対するBernoulliTSの性能の改善率（相対クリック率）
## 最後に、新ロジック(BernoulliTS)の性能と、既存ロジック(Random)の性能を比較する
## 新ロジックの性能はOPEによる推定値を、Randomの性能はログデータの目的変数の平均で推定できる真の性能を用いる
relative_policy_value_of_bernoulli_ts = estimated_policy_value["ipw"] / bandit_feedback["reward"].mean()
## 以上のOPEによって、BernoulliTSの性能はRandomの性能を19.81%上回ると推定された
print(f"Relative policy value of BernoulliTS: {relative_policy_value_of_bernoulli_ts} 倍")
