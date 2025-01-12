# policyモジュールに実装されてる各クラスについて

## Bernoulli Thompson Sampling (BernoulliTS)

- Thompson Sampling はバンディットアルゴリズムの一種で、特に「探索」と「活用」をうまくバランスさせながらアクション（選択肢）を選ぶために使われる。
- このクラスでは、アクションごとの報酬分布に基づいて最適な選択を行う仕組みが実装されている。
- bernoulli TSは、**アクションごとの報酬がベルヌーイ分布に従う場合**(=binary変数の場合!)に使われる。

初期化メソッド `__post_init__`

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
(context-freeな方策だから、結局は、同一の行動選択分布をn_rounds, len_list分だけ繰り返すだけ??:thinking:)

```python
def compute_batch_action_dist(
    self,
    n_rounds: int = 1,
    n_sim: int = 100000,
) -> np.ndarray:
    ...
```

## BaseLogisticPolicy と、それを継承した3つのクラス

- `BaseLogisticPolicy`は、ロジスティック回帰を使った(i.e. 報酬がbinary変数の場合の)contextual banditの方策学習器の基底クラス。
- 裏側では`MiniBatchLogisticRegression`(オンライン学習用のロジスティック回帰  モデルのクラス)を持ってる。
- 基底クラスを継承した3つの子クラスがある。
  - `LogisticEpsilonGreedy`
  - `LogisticUCB`
  - `LogisticThompsonSampling`
- `__post_init__` メソッドと、`select_action` メソッドが子クラスごとに異なる。逆に、それ以外のメソッドは共通の実装を使ってるみたい。(UCBとTSって、パラメータ更新方法は共通なのか??:thinking:)

- コンストラクタの共通パラメータ
  - `dim`: コンテキストベクトルの次元数
  - `n_actions`: 選択肢の数
  - `len_list`: 推薦リストの長さ(slate size)。デフォルトは1。
  - `batch_size`: パラメータ更新時に使用するサンプルのバッチサイズ。デフォルトは1。
  - `alpha_`: オンラインロジスティック回帰モデルの事前分布のハイパーパラメータ。デフォルトは1.0。
  - `lambda_`: オンラインロジスティック回帰モデルの正則化項のハイパーパラメータ。デフォルトは1.0。
  - `random_state`: アクションのサンプリング時の乱数シード。デフォルトはNone。

- 子クラス固有のパラメータ
  - `LogisticEpsilonGreedy`クラスは、`epsilon`パラメータを持つ。
    - 探索度合いを調整するパラメータ。デフォルトは0。
  - `LogisticUCB`クラスは、`epsilon`パラメータを持つ。
    - 探索度合いを調整するパラメータ。デフォルトは0。
  - `LogisticThompsonSampling`クラスは、特に固有のパラメータを持たない。

### アクション選択ロジック: `select_action` メソッド

単一のコンテキストベクトルを受け取り、選択されたアクションを返す。

- 引数:`context`: (np.ndarray, shape=(1, dim_context)) - コンテキストベクトル
  - 配列の形状に注意! 
    - 1つのデータポイントに対するコンテキストベクトルを渡す必要がある。
    - **またdocstringを見るに、(1, dim_context) の形状で渡す必要がある**...!!:thinking:
      - ただchatGPTに聞くと、使用してるライブラリや関数によっては、問題ないかもしれない。というのも、`(dim_context,)`を渡しても多くのライブラリでは自動的に`(1, dim_context)`として解釈されるので。ただ、明示的に`(1, dim_context)`として渡す方が安定して動作するので、推奨...!!:thinking:
- 返り値: (np.ndarray, shape=(len_list,))
  - 各要素は、各ポジションで選択されたアクションのidを表す。

#### 子クラスごとの実装の詳細

`LogisticEpsilonGreedy`クラスの場合

- 1. 確率`epsilon`で一様ランダムにアクションを選択する。
- 2. 確率`1 - epsilon`で、ロジスティック回帰モデルが出力する報酬期待値の推定値が大きい順にアクションを決定的に選択する。

`LogisticUCB`クラスの場合

- 1. 各行動の報酬期待値の推定値を、ロジスティック回帰モデルから取得する。
- 2. 各行動の報酬期待値の推定値の標準誤差を計算する。
- 3. UCBスコア = 報酬期待値の推定値 + 標準誤差 で計算し、UCBスコアが大きい順にアクションを決定的に選択する。

`LogisticThompsonSampling`クラスの場合

- 1. ロジスティック回帰モデルの係数ベクトルをサンプリングして得る。
  - 期待値=現在の係数ベクトル、共分散=`self.sd()`として、多変量正規分布からサンプリングする。
- 2. サンプリングしたパラメータを使って、ロジスティック回帰モデルから報酬期待値の推定値を取得する。
- 3. 報酬期待値の推定値が大きい順にアクションを決定的に選択する。

### オフライン評価のために、行動選択の確率分布を取得したい場合はどうすればいいのか...??

- どうやら、context-freeなbanditモデルにおける`compute_batch_action_dist`メソッドのような、コンテキストを受け取って行動選択確率を返すメソッドは実装されていないみたい。
  - `sample_action`メソッドでは、サンプリングした結果を返してしまっているので。

- utilsモジュールに`obp.utils.convert_to_action_dist`という関数があった。これを使えば、行動選択確率に変換できるっぽい。決定的な方策をオフライン評価するために使われてそう。
  - 参考: https://zr-obp.readthedocs.io/en/latest/_autosummary/obp.utils.html
  - 引数:
    - `n_actions`: 選択肢の数
    - `selected_actions`: 各roundで、評価方策によって選択されたアクションのリスト。
      - shape=(n_rounds, len_list)
  - 返り値: `action_dist`
    - shape=(n_rounds, n_actions, len_list) の3次元配列。
    - 各アクションの選択確率分布を表す。（決定的方策の分布になるはず）

## IPWLearner: IPWを使ったオフ方策学習器

- 特徴
  - オフラインバンディットデータ (context, action, reward, pscore...) が与えられたとき...
    - 報酬 r と行動が選択される確率 pscore を使って、目的関数を最大化するような方策を学習する。
    - 実装としては**重み付きの分類問題**として、既存の base_classifier（ロジスティック回帰とか）をトレーニングして、方策を近似。
  - 結果、「この IPWLearner の .fit() を呼び出すと、指定した base_classifier を報酬重み付きで学習して、**行動を出力する方策**が得られる」という流れ。

### fitメソッドについて

メソッドの目的: バンディットフィードバックデータを使って、IPWを使った方策を学習する。

メソッドの流れ:
1. 引数のバリデーション
    - 「rewardに負の値が含まれてないか」とか
    - 「**pscoreが与えられてない場合は、一様ランダムな選択と仮定する**」とか
      - (なるほど! じゃあnaiveな目的関数で学習をさせたい場合は、IPWLearnerでpscoreを渡さなければ良いんだ...!!:thinking:)
    - 「`len_list > 1`の場合は、`position`が必須である」とか。
      - (ex. 3つのポジションが混在してるデータなら、どのログがポジション1, 2, 3に対応するかを指定する必要がある)
2. ポジションごとにデータを分割して、**ポジションごとの行動分類器が学習**される。
   - (このクラスは、**行動候補数の分類モデルを作ってる**感じっぽい! なので新しいアイテムが追加されるためにモデルを作り直す必要がある。なので、**このIPWLearnerクラス自体はあくまで実験用のクラス**という印象...!!:thinking:)

### predictメソッドについて

- メソッドの目的: 
  - 新しいコンテキストが来たとき、各ポジションの分類器で「どの行動を選ぶか」予測して、それを (n_rounds, n_actions, len_list) 形状の3次元配列で返してる。
  - 「サンプル i のポジション p では行動 a を選びます」みたいにワンホット表現で出してる感じ。
  - 実質的に predict(context) は 「**もっともスコアの高いクラスを1つ返す**」 ので、(n_rounds,) のベクターが返ってくるんだけど、それを one-hot 的に展開して action_dist って形にしてる。

```python
def predict(self, context: np.ndarray) -> np.ndarray:
    ...
    action_dist = np.zeros((n_rounds, self.n_actions, self.len_list))
    for p in np.arange(self.len_list):
        predicted_actions_at_position = self.base_classifier_list[p].predict(context)
        action_dist[
            np.arange(n_rounds),
            predicted_actions_at_position,
            np.ones(n_rounds, dtype=int) * p,
        ] += 1
    return action_dist
```

### predict_scoreメソッドについて

- メソッドの目的:
  - .predict_proba を使って「ある行動を取る確率 (クラス確率)」を推定して、その確率を (n, n_actions, len_list) の3次元配列で返す。
  - これを使うと、実際の「確率的にアクションを選ぶ」みたいな振る舞いもできる(次の sample_action で応用している)。

### sample_action メソッド

- メソッドの目的: 
  - Gumbel-Softmax trick を使って、重複なしの「ランキング (順列)」をサンプルする。
    - i.e. Plackett-Luce 的なランキング!
  - 最終的に (n, n_actions, len_list) の形で返して、(i, action, position)に1が入ってると「サンプル i ではポジション position で行動 action を選択した」ということになる。

### predict_proba メソッド

- メソッドの目的: 
  - len_list == 1 の場合限定で、「単一アクションをソフトマックスで確率選択する」確率分布を返す感じ。
  - 「レコメンド枠が1つしかないなら、ソフトマックスによる確率分布を作れるよね」ってやつ。len_list=1 じゃないと1次元じゃなくなるんで不可。

```python
def predict_proba(
    self,
    context: np.ndarray,
    tau: Union[int, float] = 1.0,
) -> np.ndarray:
    ...
    score_predicted = self.predict_score(context=context)
    choice_prob = softmax(score_predicted / tau, axis=1)
    return choice_prob
```


## NNPolicyLearner: NNを使ったオフ方策学習器

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
    - ちなみにデフォルト値はNoneで、
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

### __post_init__ メソッドについて: NNの構造はどうなる??

- この実装では、 `hidden_layer_size` に指定した各サイズの隠れ層を積み上げて、最終的に (`出力次元 = self.n_actions`) の Linear 層と Softmax 層で終わる 多層パーセプトロン(MLP) が nn.Sequential で構築される。
  - ちなみに `nn.Sequential` は、単にPytorchでNNの各レイヤーを順番に並べて実行してくれる仕組み。


### fitメソッド

- メソッドの目的:
  - 収集されたバンディットフィードバックデータ(context, action, rewardなど)を元に、NNを使った方策を学習する。
- 引数:
  - `context`: コンテキストベクトルの配列
  - `action`: 選択されたアクションの配列
  - `reward`: 得られた報酬の配列
  - `pscore`: アクションが選択される確率の配列(Optional)
    - **pscoreが与えられない場合は、データ収集方策が一様ランダム選択だと仮定される！**
      - なるほど、じゃあnaive推定量で学習させたい場合は、pscoreを渡さなければ良いんだ...!!:thinking:
  - `position`: ポジション情報の配列(Optional)
    - 例えば、3つのポジションが混在してるデータなら、どのログがポジション1, 2, 3に対応するかを指定する必要がある。
- ざっくりやってること:
  1. 入力データのバリデーション: 入力次元数やpscoreの範囲など、オフラインバンディットデータの整合性を確認。
  2. 必要ならQ関数推定器を学習 (期待報酬の推定モデルのこと!!)
     - たとえば off_policy_objective が dr(Doubly Robust) や dm(Direct Method) のときは、まず Q関数推定器 (self.q_func_estimator) を学習して、後で使う。
       - (Q関数というのは、$\hat{q}(x, a) = \hat{E}[r | x, a]$ のような形で、コンテキストとアクションを入力として、報酬期待値の推定値を出力する関数のこと!:thinking:)
  3. オプティマイザの設定
  4. trainingデータとvalidationデータをDataLoaderに分割
  5. 反復学習して、以下を繰り返す
     1. ミニバッチ(x,a,r,pscore, position)を取得
     2. ネットワークの出力 `pi =self.nn_model(x)` を取得
     3. _estimate_policy_gradient で「方策の勾配に相当するもの」を計算 → policy_grad_arr。
     4. _estimate_policy_constraint で「方策に対する制約（行動が確率的にちゃんと割り当てられてるかなど？）」を計算。
     5. 損失関数を計算。
        - ここで、「損失関数 = -(policy_grad_arr - lambda_ * policy_constraint)」 という形式。
     6. loss.backward() → optimizer.step() でパラメータ更新。
     7. アーリーストッピング (early_stopping) の仕組みで、ある程度損失が変化しなくなったら止める。
  6. バリデーションデータでも同様に損失を計算して、改善がなければ停止 (early stopping)。

損失関数の計算についてメモ

```python
# 方策勾配の推定値を最大化したいので、勾配降下法で最小化するためにマイナスをかける
loss = -policy_grad_arr.mean() 
loss += self.policy_reg_param * policy_constraint
loss += self.var_reg_param * torch.var(policy_grad_arr)
# \nabla_{\theta}の部分は、PyTorchの自動微分機能によって暗黙的に処理される
```

### _estimate_policy_gradient メソッドについて

- メソッドの目的: 方策勾配の推定。

実装の詳細(IPWの場合):
(「反実仮想機械学習」で書いてあった通りの計算をしてそう...!!:thinking:)

```python
# 重要度重み(w_i)の計算
iw = current_pi[idx_tensor, action] / pscore
# 観測された報酬を、重要度重みで補正 (w_i * r_i)
estimated_policy_grad_arr = iw * reward
# log(\pi(a|x)) を掛ける(w_i * r_i * log(\pi(a_i|x_i)))
estimated_policy_grad_arr *= log_prob[idx_tensor, action]
```

この値に、$\nabla_{\theta}$ を適用したら、方策勾配の推定値になるはず...!

### _estimate_policy_constraint メソッドについて

- メソッドの目的: 方策学習における制約項を計算する。
  - なんか、重要度重みが大きくなりすぎないようにするための制約項っぽい...??:thinking:
- この制約項って、方策学習にどんな影響を与える??
  - データ収集方策と大きく異なるような、方策に学習されにくくなる??:thinking:

実装の詳細:

```python
idx_tensor = torch.arange(action.shape[0], dtype=torch.long)
iw = action_dist[idx_tensor, action, 0] / pscore

return torch.log(iw.mean())
```

### predict_proba メソッドについて

- メソッドの振る舞い:
  - 1. ニューラルネットワーク(self.nn_model)の出力を確率として取得
  - 2. その確率を (n_rounds, n_actions, 1) の形に変換して返す。
- メモ:
  - `nn_model(x)` は、入力 x (コンテキスト) を受け取って、**各アクションに対応するソフトマックス出力 (確率分布) を返す想定**。だから y の形状は (n_rounds, n_actions) になってるはず。
    - なるほど?? じゃあNNの出力層の数は「アクション数」になってるのか! **じゃあこれも新しいアイテムが追加されるたびにモデルを作り直す必要があるわけなので、実運用でそのまま使うのはやりづらそう**。あくまで実験用、という印象:thinking:
    - 実運用の際は、既存実装を参考に、**入力層でcontextとaction contextの両方を入力して、出力層のところで嗜好度スコアを出力する**ように変更すれば良さそう...!:thinking:

```python
def predict_proba(
    self,
    context: np.ndarray,
) -> np.ndarray:
    ...
    x = torch.from_numpy(context).float()
    y = self.nn_model(x).detach().numpy()
    return y[:, :, np.newaxis]
``` 

### sample_action メソッドについて

- メソッドの目的
  - Gumbel Softmax trick (Plackett-Luce) を使って、重複なしのランキングを確率的にサンプリングする。
  - `predict_proba` で出力された確率分布を使って、ランキングをサンプリングする感じ。

祝新年:)
2024年の目標は存在感でした！https://x.com/moritama7431/status/1741602342947205614
数回のブログや発表機会などを通じて、以前よりも、推薦やMLOps周りに興味持ってる自分の存在を発信できた気がします!(素敵な機会に感謝...!)
2025年の目標は、存在感と、ビジネスでMLの成果をスケールさせる取り組みを頑張る...!

2024年の目標は存在感でした！https://x.com/moritama7431/status/1741602342947205614
数回のブログや発表機会などを通じて、推薦やMLOps周りに興味持ってる自分の存在を、以前よりも発信できた気がします!(素敵な機会に感謝...!)
