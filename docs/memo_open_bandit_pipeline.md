## 参考資料

- OBPの活用例: <https://github.com/st-tech/zr-obp/tree/master/examples>
- usaitoさんのcontexual banditの記事
  - part1: <https://qiita.com/usaito/items/e727dcac7325b50d4d4c#%E3%81%AF%E3%81%98%E3%82%81%E3%81%AB>
  - part2: <https://qiita.com/usaito/items/46a9af98c6f6cace05c4>

## 合成データセットクラス `SyntheticBanditDataset` のメモ

- 合成バンディットデータ(方策評価やバンディットアルゴリズムの評価に使われるデータ)を生成するためのクラス
- 主なパラメータ
  - `n_actions`: int
    - アクション数
  - `dim_context` (int, default=1)
    - 特徴量の次元数
  - `reward_type` (str, default="binary")
    - 報酬の種類。`binary`または`continuous`を選択。
      - あ、これでMECEになるのか...! 確かにA/Bテストのmetricも、binary metricかnon-binary metricかで分類してるし...!:thinking:
    - binaryの場合、報酬はベルヌーイ分布からサンプリングされる。成功/失敗などのbinaryな報酬に使う
    - continuousの場合、報酬は正規分布からサンプリングされる
  - `reward_function` (Callable[[np.ndarray, np.ndarray], np.ndarray], default=None)
    - 各アクションと文脈の組み合わせに対する期待報酬、を返す関数。
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
  - `action_context` (np.ndarray, default=None)
    - アクションを特徴づけるためのベクトル表現。
    - 指定しない場合は、各アクションはone-hot表現で表される(アクション特徴量を持たない)。
      - (これはMIPS推定量などのための指定かな...!:thinking:)
  - `behavior_policy_function` (Callable[[np.ndarray, np.ndarray], np.ndarray], default=None)
    - データ収集方策(logging policy, behavior policy)を表す関数。
      - 具体的には、contextとaction特徴量を入力として受け取り、各アクションのlogit values(=要するに意思決定に利用するスコア的なスカラー値)のベクトルを返す関数。
    - 指定しない場合、期待報酬にソフトマックスを適用した値が、データ収集方策として利用される。
      - (i.e. contextを利用せずに、単にE[r|a]が大きいアクションがより選ばれやすくなる方策)
      - この場合 `beta`引数で逆温度パラメータを指定できる。
  - `beta` (int or float, default=1.0)
    - データ収集方策で、ソフトマックス関数でp_0(a|x)を計算するための逆温度パラメータ。
      - 小さいとランダム性が高まり、大きいとgreedyになる。
      - また、負の値を指定すると非最適な方策になる。(スコアが低いほど選びやすくなる、みたいな:thinking:)
  - `n_deficient_actions` (int, default=0)
    - ログデータ中で選択確率がゼロになる「不足アクション」の数を指定する。
    - この値が正の場合、共通サポート仮定が成り立たなくなり、IPW推定量のバイアスにつながる可能性がある。
      - n_actions-1未満の整数である必要がある。
  - `random_state` (int, default=12345)
    - 乱数シードを設定し、合成データの再現性を確保する。
  - `dataset_name` (str, default=`synthetic_bandit_dataset`)
    - データセットの名前。主にデータ管理やロギング用途で使用される。

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

- `obtain_batch_bandit_feedback`
- `calc_ground_truth_policy_value`
