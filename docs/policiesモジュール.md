# policyモジュールに実装されてる各クラスについて

## Bernoulli Thompson Sampling (BernoulliTS)

- Thompson Sampling はバンディットアルゴリズムの一種で、特に「探索」と「活用」をうまくバランスさせながらアクション（選択肢）を選ぶために使われる。
- このクラスでは、アクションごとの報酬分布に基づいて最適な選択を行う仕組みが実装されている。

初期化メソッド __post_init__

- `is_zozotown_prior` がFalseの場合
  - デフォルトの事前分布として alpha=1, beta=1 のベータ分布を使用する。
  - この場合、すべてのアクションが均等に選ばれる初期状態になる。

アクション選択メソッド `select_action`

```python
def select_action(self) -> np.ndarray:
    ...
```

- 目的: 現在得られてる報酬データを元に、最適なアクションを選択する。
- 実装の詳細
  - 1. Beta分布からサンプリングを行い、各アクションの報酬の期待値の推定値?をサンプリングする。
  - 2. 報酬の期待値が高い順にソートし、len_listの数だけアクションを選択する。

```python
predicted_rewards = self.random_.beta(
    a=self.reward_counts + self.alpha,
    b=(self.action_counts - self.reward_counts) + self.beta,
)
```

パラメータ更新メソッド `update_params`

```python
def update_params(self, action: int, reward: float) -> None:
    ...
```

- 目的: 選択したアクションによって得られた報酬を元に、モデル内部のパラメータを更新する。

バッチで行動選択分布を計算するメソッド `compute_batch_action_dist`

```python
def compute_batch_action_dist(
    self,
    n_rounds: int = 1,
    n_sim: int = 100000,
) -> np.ndarray:
    ...
```

## Logistic Thompson Sampling (LTS)

- 目的: コンテキスト（例えば、ユーザー情報や現在の状態）に基づいて、与えられた選択肢（アクション）の中から最適なものを選ぶ。
- 裏側では`MiniBatchLogisticRegression`(オンライン学習用のロジスティック回帰モデルのクラス)を持ってる。

### パラメータ

必須

- dim: コンテキストベクトルの次元数
- n_actions: 選択肢の数

オプショナル

- `len_list`: 推薦リストの長さ
- `batch_size`: パラメータ更新時に使用するサンプルの数。
- `random_state`: 乱数シード
- `alpha`: ロジスティック回帰の事前分布のハイパーパラメータ。
- `lambda_`: 正則化項のハイパーパラメータ。過学習を防ぐために使用。

### アクション選択ロジック

- `select_action`メソッド: コンテキストベクトルを受け取り、アクションを選択する。
-

## NNPolicyLearner: NNを使ったOff-policy学習器

- 特徴
  - 方策をニューラルネットワークで表現する。
    - 隠れ層のサイズや活性化関数などを指定できる。
  - 様々なOPE推定量を選択して、方策の勾配を推定できる。
    - Direct Method, Inverse Probability Weighting, Doubly Robustなどの手法が選択可能。
  - 方策正則化と分散正則化のパラメータを設定できる。

### 主要なパラメータ

- 基本情報
  - dim_context: コンテキストベクトルの次元数。NNの入力サイズを決める重要なパラメータ。
    - もしコンテキストと、アクションコンテキスト両方を使う場合は、たぶん両者を合計した次元数を指定するべき...!:thinking:
  - n_actions: 選択可能なアクションの数。(例えば、広告の選択肢数)
  - len_list: 推薦システムなどで同時に提案するアクション数。
- 目的関数についてのパラメータ:
  - `off_policy_objective`: 勾配計算に使用するOPE手法を指定。
    - `dm`
    - `ipw`
    - `dr`
    - その他、IPW手法の改善ver.(`snips`, `ipw-os`, `ipw-sbgauss`)
  - `lambda_`: `snips`や`ipw`系の手法で使用するハイパーパラメータ。報酬のシフトや重みの調整に利用。
- NNの構造についてのパラメータ:
  - `hidden_layer_size`: 隠れ層のサイズをタプル形式で指定（例: (100,) なら1層100ユニット）。
  - `activation`: 活性化関数（relu, tanh, など）。
  - `solver`: 最適化アルゴリズム（例: adam, sgd）。
  - `alpha`: L2正則化項の係数。
- 学習の詳細に関するパラメータ:
  - `batch_size`: ミニバッチサイズ（"auto" は200かサンプル数の多い方を選択）。
  - `learning_rate_init`: 学習率の初期値。
  - `max_iter`: 学習の最大エポック数。
  - `shuffle`: データシャッフルの有無。
  - `early_stopping`: 早期終了を有効化するかどうか。
- その他のハイパーパラメータ:
  - `momentum`: SGDでのモーメント値（加速収束に使用）。
  - `nesterovs_momentum`: ネステロフモーメントの使用有無。
  - `validation_fraction`: バリデーションデータの割合（早期終了時に使用）。
  - `beta_1` と `beta_2`: Adam オプティマイザの係数。
  - `epsilon`: 数値安定性を向上させるためのノイズ。
- Q関数推定器の設定
  - `q_func_estimator_hyperparams`: Q関数推定器のハイパーパラメータを辞書形式で指定。

### 使われる技術や関連手法

- このクラスでは、OPE手法を活用してポリシーの勾配を計算し、ニューラルネットワークを通じて最適なポリシーを学習します。
- クラスの設計は、以下のような研究に基づいています：
  - Adam 最適化アルゴリズム（Kingma and Ba, 2014）
  - ログバンディットデータでの深層学習手法（Joachims et al., 2018）
  - 縮小法を用いた二重頑健推定（Su et al., 2020）
  - サブガウス分布と微分可能な重要度サンプリング（Metelli et al., 2021）

### fitメソッド

- 収集されたバンディットフィードバックデータ(context, action, rewardなど)を元に、NNを使った方策を学習する。
