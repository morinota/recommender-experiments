# MINDデータセットを使ったニュース推薦システム MVPS

## 概要

MINDデータセット(Microsoft News Dataset)を使用したニュース推薦システムのMVPS(Minimal Viable Prediction Service)実装です。

## システム構成

### 予測問題
ユーザーが次にクリックするニュース記事を予測する二値分類問題

### KPI
クリックスルー率(CTR)の向上

### ML Proxy Metric
ニュースクリックの確率(0/1の二値分類)

## 実装されたパイプライン

### 1. Feature Pipeline - Backfill (`1_backfill_feature_groups.py`)
- MINDデータセットから特徴量を抽出してバックフィル
- ニュース特徴グループ: カテゴリ、エンティティ数など
- impression特徴グループ: ユーザー履歴長など
- 学習データの作成(news特徴とimpression特徴のjoin)

実行方法:
```bash
uv run python src/recommender_experiments/app/mind_ml_project/1_backfill_feature_groups.py
```

### 2. Feature Pipeline - Daily (`2_daily_feature_pipeline.py`)
- 新しいニュース記事とユーザー行動データの増分処理
- 既存の特徴グループに新しいデータを追加
- 学習データの再生成
- **Note**: MINDデータセットは静的なため、devデータを「新しいデータ」として扱うデモ実装

実行方法:
```bash
uv run python src/recommender_experiments/app/mind_ml_project/2_daily_feature_pipeline.py
```

### 3. Training Pipeline (`3_training_pipeline.py`)
- Gradient Boosting分類モデルの訓練
- 10万件のサンプルデータで学習(MVPS版)
- 評価メトリクス: Accuracy, ROC-AUC, Log Loss

実行方法:
```bash
uv run python src/recommender_experiments/app/mind_ml_project/3_training_pipeline.py
```

### 4. Inference Pipeline (`4_batch_inference_pipeline.py`)
- バッチ推論によるTOP-K推薦生成
- ユーザーごとの推薦ニュース生成

実行方法:
```bash
uv run python src/recommender_experiments/app/mind_ml_project/4_batch_inference_pipeline.py
```

### 5. Dashboard (`5_dashboard.py`)
- marimoを使ったインタラクティブダッシュボード
- データ概要、カテゴリ分布、ユーザー行動分析を可視化
- モデル情報と推薦結果をインタラクティブに表示

実行方法:
```bash
uv run marimo edit src/recommender_experiments/app/mind_ml_project/5_dashboard.py
```

ブラウザで `http://localhost:2718` にアクセスしてダッシュボードを確認できます。

## FTIパイプラインの実行順序

初回セットアップ:
```bash
# 1. 特徴量のバックフィル
uv run python src/recommender_experiments/app/mind_ml_project/1_backfill_feature_groups.py

# 2. モデルの訓練
uv run python src/recommender_experiments/app/mind_ml_project/3_training_pipeline.py

# 3. バッチ推論
uv run python src/recommender_experiments/app/mind_ml_project/4_batch_inference_pipeline.py

# 4. ダッシュボードで結果確認
uv run marimo edit src/recommender_experiments/app/mind_ml_project/5_dashboard.py
```

日次運用(想定):
```bash
# 1. 新しいデータで特徴量を増分更新
uv run python src/recommender_experiments/app/mind_ml_project/2_daily_feature_pipeline.py

# 2. 定期的にモデルを再訓練(例: 週次)
uv run python src/recommender_experiments/app/mind_ml_project/3_training_pipeline.py

# 3. 毎日バッチ推論を実行
uv run python src/recommender_experiments/app/mind_ml_project/4_batch_inference_pipeline.py

# 4. ダッシュボードで結果確認
uv run marimo edit src/recommender_experiments/app/mind_ml_project/5_dashboard.py
```

## データ構造

### 特徴グループ(Feature Store)
- `news_features.parquet`: ニュース記事の特徴量
- `impression_features.parquet`: impression単位の特徴量
- `training_data.parquet`: 学習データ

### モデルレジストリ
- `news_recommendation_model.pkl`: 訓練済みGradientBoostingモデル

### 推論結果
- `user_recommendations.parquet`: ユーザーごとのTOP-K推薦
- `predictions.parquet`: 全impressionの予測スコア

## 改善の余地

1. **特徴量エンジニアリング**
   - ユーザーの閲覧履歴からより高度な特徴量を生成
   - ニュース記事の埋め込みベクトル(entity_embedding.vec)の活用
   - 時系列特徴(曜日、時間帯など)の追加

2. **モデル改善**
   - XGBoostやLightGBMの利用(現在はOpenMPの問題でGradientBoostingを使用)
   - ハイパーパラメータチューニング
   - より多くのデータでの学習

3. **推論パイプライン**
   - 特徴量カラムの一致処理の改善
   - オンライン推論への対応

4. **評価**
   - オフライン評価メトリクスの追加(NDCG、MRRなど)
   - A/Bテスト設計

## テスト

```bash
# 全テスト実行
uv run pytest tests/test_mind_ml_project/ -v

# 特定のテストのみ
uv run pytest tests/test_mind_ml_project/test_data_loader.py -v
uv run pytest tests/test_mind_ml_project/test_feature_functions.py -v
```

## 参考資料

- [Building ML Systems with a Feature Store (O'Reilly)](https://www.oreilly.com/library/view/building-ml-systems/9781098157234/)
  - Chapter 2: Machine Learning Pipelines
  - Chapter 3: Your Friendly Neighborhood Air Quality Forecasting Service
- [MIND Dataset (Microsoft)](https://msnews.github.io/)
