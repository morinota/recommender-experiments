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
