## 参考資料

- OBPの活用例: <https://github.com/st-tech/zr-obp/tree/master/examples>
- usaitoさんのcontexual banditの記事
  - part1: <https://qiita.com/usaito/items/e727dcac7325b50d4d4c#%E3%81%AF%E3%81%98%E3%82%81%E3%81%AB>
  - part2: <https://qiita.com/usaito/items/46a9af98c6f6cace05c4>

## 合成データセットクラス `SyntheticBanditDataset` のメモ

- 合成バンディットデータ(方策評価やバンディットアルゴリズムの評価に使われるデータ)を生成するためのクラス

### 主な引数

- `n_actions`: int
  - アクション数
- `dim_context` (int, default=1)
  - 特徴量の次元数

報酬の期待値の設定に関する引数たち

- `reward_type` (str, default="binary")
  - 報酬の種類。`binary`または`continuous`を選択。
    - あ、これでMECEになるのか...! 確かにA/Bテストのmetricも、binary metricかnon-binary metricかで分類してるし...!:thinking:
  - binaryの場合、報酬はベルヌーイ分布からサンプリングされる。成功/失敗などのbinaryな報酬に使う
  - continuousの場合、報酬は正規分布からサンプリングされる
- `reward_function` (Callable[[np.ndarray, np.ndarray], np.ndarray], default=None)
  - 各アクションと文脈の組み合わせに対する期待報酬の**真の値**、を返す関数。
    - 数式で表すと $q(x, a) = E[r|x, a]$
  - 指定しない場合は、contextに依存しない一様分布に基づく期待報酬が自動的にサンプリングされる
    - 期待報酬関数 $q(x, a) = q(a) = E[r|a]$ とみなして、各アクションに対する期待報酬が一様分布からサンプリングされるってことか...!:thinking:
- `reward_std` (float, default=1.0)
  - 報酬分布の標準偏差。この値が大きいほど、報酬のばらつきが大きくなる。
    - Normal(q(x, a), reward_std) から報酬をサンプリングするってことだよね...!:thinking:
  - このパラメータはreward_type="continuous"の時にのみ有効。binaryの場合、報酬分布のパラメータは期待値 q(x, a) のみなので不要...!:thinking:
- `reward_noise_distribution` (str, default="normal")
  - 報酬に加えるノイズの分布。`normal`または`truncated_normal`を選択。
  - `truncated_normal`を選択した場合は、報酬は正規分布からサンプリングされるが、その値が0未満の場合は0にクリップされる。
  - このパラメータもreward_type="continuous"の時にのみ有効

アイテム特徴量に関する引数

- `action_context` (np.ndarray, default=None)
  - アクションを特徴づけるためのベクトル表現。
  - 指定しない場合は、各アクションはone-hot表現で表される(アクション特徴量を持たない)。
    - (これはMIPS推定量などのための指定かな...!:thinking:)

データ収集方策(logging policy, behavior policy)に関する引数

- `behavior_policy_function` (Callable[[np.ndarray, np.ndarray], np.ndarray], default=None)
  - データ収集方策(logging policy, behavior policy)を表す関数。
    - 具体的には、contextとaction特徴量を入力として受け取り、各アクションのlogit values(=要するに意思決定に利用するスコア的なスカラー値)のベクトルを返す関数。
    - (実際にサンプリングする際は、ソフトマックス変換をかませてp_0(a|x)とするっぽい:thinking:)
      - アクション特徴量を使わない場合は、関数の引数としては受け取るが何も使わなければOKのはず...!:thinking:
  - 指定しない場合、期待報酬にソフトマックスを適用した値が、データ収集方策として利用される。
    - (i.e. contextを利用せずに、単にE[r|a]が大きいアクションがより選ばれやすくなる方策)
    - この場合 `beta`引数で逆温度パラメータを指定できる。
- `beta` (int or float, default=1.0)
  - データ収集方策で、ソフトマックス関数でp_0(a|x)を計算するための逆温度パラメータ。
    - 小さいとランダム性が高まり、大きいとgreedyになる。
    - また、負の値を指定すると非最適な方策になる。(スコアが低いほど選びやすくなる、みたいな:thinking:)
  - これ適用しないようにしたいなぁ...:thinking:

シミュレーションするバンディットデータに関する引数

- `n_deficient_actions` (int, default=0)
  - ログデータ中で選択確率がゼロになる「不足アクション」の数を指定する。
  - この値が正の場合、共通サポート仮定が成り立たなくなり、IPW推定量のバイアスにつながる可能性がある。
    - n_actions-1未満の整数である必要がある。
- `random_state` (int, default=12345)
  - 乱数シードを設定し、合成データの再現性を確保する。
- `dataset_name` (str, default=`synthetic_bandit_dataset`)
  - データセットの名前。主にデータ管理やロギング用途で使用される。

### 期待報酬やデータ収集方策のカスタム例

reward_function (期待報酬の真の値を返す関数)の指定例
(三つの引数をとる必要があるっぽい: context, action_context, random_state)

```python
def expected_reward_function(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: int = None,  # この引数を追加
) -> np.ndarray:
    """(アクションa, 文脈x)の各組み合わせに対する期待報酬 E[r|x,a] を定義する関数
    今回の場合は、推薦候補4つの記事を送った場合の報酬rの期待値を、文脈xに依存しない固定値として設定する
    ニュース1: 0.2, ニュース2: 0.15, ニュース3: 0.1, ニュース4: 0.05
    """
    n_rounds = context.shape[0]
    n_actions = action_context.shape[0]

    # 固定の期待報酬を設定 (n_actions=4として固定値を設定)
    fixed_rewards = np.array([0.2, 0.15, 0.1, 0.05])

    # 文脈の数だけ期待報酬を繰り返して返す
    return np.tile(fixed_rewards, (n_rounds, 1))
```

behavior_policy_function引数の指定の例

```python
def custom_behavior_policy(context: np.ndarray, action_context: np.ndarray) -> np.ndarray:
    bias = np.array([1, 0.5, -0.5, -1, 0.2])  # 任意のバイアス
    logits = np.dot(context, action_context.T) + bias  # バイアスを追加
    return logits
  
dataset = SyntheticBanditDataset(
    n_actions=5,
    dim_context=3,
    reward_function=logistic_reward_function,
    behavior_policy_function=custom_behavior_policy,  # 指定する
    random_state=12345
)
```

### 主なメソッド

#### `obtain_batch_bandit_feedback`

- バッチ単位で、バンディットフィードバックデータを生成する。
  - 具体的には、`n_rounds`引数のラウンド数分、(コンテキスト、期待報酬、行動ポリシー、実際のアクション、報酬)を生成して、バッチデータとして返す。
- `behavior_policy_function`が指定されてる場合は、その関数を使用してスコアを生成し、ソフトマックス変換をしてデータ収集方策を定義。
  - 指定されていない場合は、期待報酬をもとにしたソフトマックス変換でデータ収集方策を定義。
- 生成されたデータは以下のkey-valueのdictとして返される。
  - n_rounds: ラウンド数
  - n_actions: アクション数
  - context: コンテキスト
  - action_context: アクション特徴量
  - action: 実際に選択されたアクション
  - position: ポジション (推薦リストにおける表示位置。アイテムを1つ推薦する場合はNone)
  - reward: 報酬
  - expected_reward: 期待報酬
  - pi_b: データ収集方策が文脈xを受け取った場合にアクションaを選択する確率 P(a|x) の配列
  - pscore: 選択されたactionに対するデータ収集方策の傾向スコア(propensity score)の配列

pi_bメソッドの算出方法についてメモ

- データ収集方策(`behavior_policy_function`引数)の値を元に、`beta`引数に基づいてソフトマックス変換をかけて、確率分布の値を返す。
  
```python
pi_b = softmax(self.beta * pi_b_logits)
# pi_b_logits: データ収集方策が返す値
```

- この仕様はない方が都合いいなぁ...:thinking:
  - 仕方なく、暫定的に`SyntheticBanditDataset`の内部コードを一行変更して、softmax変換をかけないようにしてみた...:thinking:

```python
pi_b_logits = self.behavior_policy_function(
                context=contexts,
                action_context=self.action_context,
                random_state=self.random_state,
            )
...
# pi_b = softmax(self.beta * pi_b_logits)
pi_b = pi_b_logits
```

#### `calc_ground_truth_policy_value`

- 引数として、`expected_reward`(=各(context, action)ペアに対する期待報酬 E[r|a,x]。合成データなら既知なので)と、`action_dist`(=評価方策の行動選択確率) を受け取り、評価方策の累積報酬の期待値を計算する。

```python
dataset.calc_ground_truth_policy_value(
            # 擬似データの各(context, action)ペアに対する期待報酬 E[r|a,x] の一覧
            expected_reward=bandit_feedback_test["expected_reward"],
            # 評価方策の行動選択確率 P(a|x) 
            action_dist=action_dist,
        )
```

### reward_function として指定される例に挙げられる logistic_reward_function について

- この関数は、contextとactionを受け取りその組み合わせに対する期待報酬の真の値を返すものであり、ロジスティック回帰に基づいた報酬生成プロセスをシミュレートしている。
- 期待報酬の計算プロセス
  - 1. _base_reward_functionを使って、文脈x、アクションa、および文脈*アクションの相互作用を元にlogit値を計算する。
    - $logit(x,a) = w_c * x + w_a * a + w_{ca} * (x * a)$
    - 重み $w_c, w_a, w_{ca}$ は適当にランダムに設定されるっぽい!:thinking:
  - 2. シグモイド関数によってlogit値を0~1の範囲に変換する。
    - $q(x,a) = \sigma(logit(x,a)) = \frac{1}{1 + \exp(-logit(x,a))}$
  - 3. 期待報酬の真の値 E[r|x,a] として返却。

## 線形回帰を用いたcontextual banditクラスたちについてメモ

obp.policyには、以下の線形回帰を用いたコンテキストバンディットのアルゴリズム達が実装されている。

### 　ベースクラス: BaseLinPolicy

## オフライン評価用クラス `OffPolicyEvaluation` のメモ

- このクラスの主な役割: 複数のOPE推定量を用いて、評価方策の性能を同時に評価すること。
- 主なメソッド
  - `estimate_policy_values()`: 各OPE推定量を用いて、評価方策の性能(累積報酬の期待値)を推定する。
  - `

### estimate_policy_values メソッドについて

- パラメータ
  - `action_dist`:
    - 評価方策の行動選択確率 P(a|x) の配列。
    - 形状は (n_rounds, n_actions, len_list) である必要がある。
    - 各ラウンドで、評価方策がどのように行動を選択するかを表す。
- オプションパラメータ
  - `estimated_rewards_by_reg_model`:
    - 回帰モデルによって推定された各アクションの期待報酬 $\hat{q}(x,a)$ の配列。
    - 形状は (n_rounds, n_actions, len_list) である必要がある。
    - モデル依存型のOPE推定量（DM, DR）では必須。
  - `estimated_pscore`:
    - データ収集方策の傾向スコアの推定値 $\hat{\pi}_b(a|x)$ の配列。
    - 形状は (n_rounds,)である必要がある。
    - IPS(IPW)推定量などを計算する際に必要。
    - default=Noneであり、指定しない場合もIPS推定量はエラーなく暗黙的にnaive推定量として計算されてしまう?? :thinking:
  - `estimated_importance_weights`
    - 重要度重みの推定値 $\hat{w}(x,a)$ の配列。
    - 形状は (n_rounds, ) もしくは、dict[str, array-like] である必要がある。
    - obp.ope.ImportanceWeightEstimator`によって実装された教師あり分類によって推定された重要度の重み。
    - estimated_pscoreがすでに指定されている場合は不要っポイ??:thinking:
    - array-likeで指定されると、全てのOPE推定量に適用される。
    - dictで指定されると、各OPE推定量に対して個別に適用される。

  - `action_embed`:
  - `pi_b`:
  - `p_e_a`:
