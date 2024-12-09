
## 実行コマンド

以下のコマンドでシミュレーションを実行します。

```bash
poetry run python -m src.recommender_experiments.app.ope_experiment \
    --n-runs 10 \
    --n-rounds 10000 \
    --n-actions 10 \
    --dim-context 5 \
    --beta 3.0 \
    --base-model-for-evaluation-policy logistic_regression \
    --base-model-for-reg-model random_forest \
    --n-jobs 4 \
    --random-state 12345
```
