"""MINDニュース推薦システム ダッシュボード.

marimoを使ったインタラクティブなダッシュボードで、
推薦結果やモデル性能を可視化します。

実行方法:
    uv run marimo edit src/recommender_experiments/app/mind_ml_project/5_dashboard.py
"""

import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # MINDニュース推薦システム ダッシュボード

    このダッシュボードでは、MINDデータセットを使ったニュース推薦システムの結果を可視化します。
    """)
    return


@app.cell
def _():
    import pickle
    from pathlib import Path

    import polars as pl

    from recommender_experiments.app.mind_ml_project.utils.config import Config
    from recommender_experiments.app.mind_ml_project.utils.data_loader import (
        load_behaviors_df,
        load_news_df,
    )
    return Config, load_behaviors_df, load_news_df, pickle, pl


@app.cell
def _(mo):
    mo.md("""
    ## 1. データ概要
    """)
    return


@app.cell
def _(Config, load_behaviors_df, load_news_df, pl):
    # データ読み込み
    news_df = load_news_df(Config.MIND_TRAIN_DIR / "news.tsv")
    behaviors_df = load_behaviors_df(Config.MIND_TRAIN_DIR / "behaviors.tsv")

    # 特徴量ストアのデータ
    news_features_df = pl.read_parquet(Config.FEATURE_STORE_DIR / "news_features.parquet")
    impression_features_df = pl.read_parquet(Config.FEATURE_STORE_DIR / "impression_features.parquet")

    # データサマリー
    data_summary = {
        "ニュース記事数": news_features_df.shape[0],
        "ユニークユーザー数": impression_features_df["user_id"].n_unique(),
        "総impression数": impression_features_df.shape[0],
        "クリック数": impression_features_df.filter(pl.col("clicked") == 1).shape[0],
        "CTR": f"{impression_features_df.filter(pl.col('clicked') == 1).shape[0] / impression_features_df.shape[0] * 100:.2f}%",
    }
    return data_summary, impression_features_df, news_df, news_features_df


@app.cell
def _(data_summary, mo, pl):
    # データサマリーをテーブル表示
    summary_df = pl.DataFrame({"メトリクス": list(data_summary.keys()), "値": [str(v) for v in data_summary.values()]})

    mo.ui.table(summary_df, selection=None)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. ニュースカテゴリ分布
    """)
    return


@app.cell
def _(news_features_df, pl):
    # カテゴリごとのニュース記事数
    category_counts = (
        news_features_df.group_by("category").agg(pl.len().alias("count")).sort("count", descending=True)
    )
    return (category_counts,)


@app.cell
def _(category_counts, mo):
    import plotly.express as px

    # 棒グラフで可視化
    fig_category = px.bar(
        category_counts.to_pandas(),
        x="category",
        y="count",
        title="カテゴリ別ニュース記事数",
        labels={"category": "カテゴリ", "count": "記事数"},
    )

    mo.ui.plotly(fig_category)
    return (px,)


@app.cell
def _(mo):
    mo.md("""
    ## 3. ユーザー行動の分析
    """)
    return


@app.cell
def _(impression_features_df, pl):
    # ユーザーごとのクリック数と履歴長の分布
    user_stats = (
        impression_features_df.group_by("user_id")
        .agg([pl.col("clicked").sum().alias("total_clicks"), pl.col("history_length").first().alias("history_length")])
        .sort("total_clicks", descending=True)
    )
    return (user_stats,)


@app.cell
def _(mo, px, user_stats):
    # ヒストグラムで可視化
    fig_clicks = px.histogram(
        user_stats.to_pandas(),
        x="total_clicks",
        nbins=50,
        title="ユーザーごとのクリック数分布",
        labels={"total_clicks": "クリック数", "count": "ユーザー数"},
    )

    mo.ui.plotly(fig_clicks)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. モデル性能
    """)
    return


@app.cell
def _(Config, pickle):
    # モデル読み込み
    model_path = Config.MODEL_REGISTRY_DIR / "news_recommendation_model.pkl"

    if model_path.exists():
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        model_info = {
            "モデルタイプ": type(model).__name__,
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "learning_rate": model.learning_rate,
        }
    else:
        model_info = {"エラー": "モデルが見つかりません"}
    return (model_info,)


@app.cell
def _(mo, model_info, pl):
    # モデル情報をテーブル表示
    model_df = pl.DataFrame({"パラメータ": list(model_info.keys()), "値": [str(v) for v in model_info.values()]})

    mo.ui.table(model_df, selection=None)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. 推薦結果サンプル
    """)
    return


@app.cell
def _(Config, mo, pl):
    # 推薦結果の読み込み
    recommendations_path = Config.INFERENCE_OUTPUT_DIR / "user_recommendations.parquet"

    if recommendations_path.exists():
        recommendations_df = pl.read_parquet(recommendations_path)

        # ユーザー選択用のドロップダウン
        user_ids = recommendations_df["user_id"].to_list()
        user_selector = mo.ui.dropdown(options=user_ids, label="ユーザーを選択", value=user_ids[0] if user_ids else None)

        user_selector
    else:
        mo.md("**推薦結果が見つかりません。先に推論パイプラインを実行してください。**")
        user_selector = None
        recommendations_df = None
    return recommendations_df, user_selector


@app.cell
def _(mo, news_df, pl, recommendations_df, user_selector):
    # 選択されたユーザーの推薦ニュース表示
    if user_selector is not None and user_selector.value is not None:
        user_recommendations = recommendations_df.filter(pl.col("user_id") == user_selector.value)

        if user_recommendations.shape[0] > 0:
            recommended_news_ids = user_recommendations["recommended_news"][0]
            scores = user_recommendations["scores"][0]

            # 推薦されたニュース記事の詳細を取得
            recommended_news_detail = news_df.filter(pl.col("news_id").is_in(recommended_news_ids)).select(
                ["news_id", "category", "subcategory", "title"]
            )

            # スコアを追加
            recommended_news_detail = recommended_news_detail.with_columns(
                pl.Series("score", scores[: recommended_news_detail.shape[0]])
            ).sort("score", descending=True)

            mo.ui.table(recommended_news_detail, label=f"ユーザー {user_selector.value} への推薦")
        else:
            mo.md(f"**ユーザー {user_selector.value} の推薦結果が見つかりません**")
    else:
        mo.md("")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 次のステップ

    - モデルの再訓練でより多くのデータを使用
    - ハイパーパラメータチューニング
    - より高度な特徴量エンジニアリング
    - オンライン推論への対応
    """)
    return


if __name__ == "__main__":
    app.run()
