# ecommerce-forecasting-recommender
Production-ready MLOps platform for e-commerce demand forecasting and personalised product recommendations using the Olist dataset, with DVC versioning, MLflow experiment tracking, Airflow orchestration, FastAPI serving, CI/CD, and drift monitoring.

Explore [the simple baselines notebook](notebooks/simple_baselines.ipynb) for worked examples, forecasting and recommendation evaluation, and local MLflow experiment logging. Run the data preparation and forecasting feature scripts first, then execute the notebook from top to bottom using the project environment with `ipykernel` installed. Results and the MLflow database are written to `data/baselines/`.

Install the Python 3.12 dependencies for all scripts, pipelines, tests and notebooks:

```bash
python -m pip install -r requirements.txt
```

For notebook-only work, `notebooks/requirements.txt` provides a smaller subset.

## Command-line scripts

Scripts are grouped by pipeline stage and call the corresponding implementation in `src/`:

```text
scripts/
├── data/
│   ├── download_olist_data.py
│   ├── validate_olist_data.py
│   ├── clean_olist_data.py
│   └── prepare_curated_data.py
├── features/
│   ├── build_forecasting_features.py
│   └── build_recommendation_features.py
├── models/
│   ├── baseline/
│   │   ├── forecasting_baselines.py
│   │   └── recommendation_baselines.py
│   └── production/
│       ├── validate_inputs.py
│       ├── train_forecasting.py
│       └── train_recommendation.py
└── maintenance/
    ├── cleanup_data.py
    └── reset_repository.py
```

Run commands from the repository root using the new paths:

```bash
.venv/bin/python scripts/data/download_olist_data.py
.venv/bin/python scripts/data/validate_olist_data.py
.venv/bin/python scripts/data/clean_olist_data.py
.venv/bin/python scripts/data/prepare_curated_data.py
.venv/bin/python scripts/features/build_forecasting_features.py
.venv/bin/python scripts/models/baseline/forecasting_baselines.py
```

For recommendation features, supply a customer, state and historical cutoff:

```bash
.venv/bin/python scripts/features/build_recommendation_features.py \
  --customer-id CUSTOMER_UNIQUE_ID --state SP --as-of 2018-08-01
.venv/bin/python scripts/models/baseline/recommendation_baselines.py --help
```

Use `--help` on any script to inspect its arguments. Import reusable functions from `src.data`, `src.features` or `src.models.baseline` in application code and notebooks.

## Run the complete pipeline with Airflow

The DAG in `pipelines/data_pipeline.py` is named `olist_end_to_end_pipeline` and runs:

```text
download_olist_data
  → validate_olist_data
  → clean_olist_data
  → prepare_curated_data
  → build_forecasting_features
  → build_recommendation_features
  → run_forecasting_baselines
  → run_recommendation_baselines
```

It is manually triggered, with one active run and one task at a time. Later tasks run only after their dependencies succeed. Complete raw downloads are reused; baseline retries can create additional MLflow runs.

Install `requirements.txt` in `.venv` and configure Kaggle authentication for the Airflow worker using `~/.kaggle/kaggle.json` or `KAGGLE_API_TOKEN`. The worker must have the Kaggle CLI and all project dependencies. Start a local development instance from the repository root:


```bash
source .venv/bin/activate

# Use a fresh, project-specific Airflow home.
export AIRFLOW_HOME="$PWD/airflow"
export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN="sqlite:///$AIRFLOW_HOME/airflow.db"

# Load only this project's DAGs.
export AIRFLOW__CORE__DAGS_FOLDER="$PWD/pipelines"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
```
```bash
export AIRFLOW_HOME="$PWD/airflow"
export AIRFLOW__CORE__DAGS_FOLDER="$PWD/pipelines"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
.venv/bin/airflow standalone
```

Open the address printed by Airflow `http://localhost:8080.`, unpause `olist_end_to_end_pipeline` and trigger it. In another terminal with the same environment variables, you can also trigger it with:

```bash
.venv/bin/airflow dags trigger olist_end_to_end_pipeline
```

The trigger form exposes `validation_days`, `test_days`, `recommendation_customer_id`, `recommendation_state`, `recommendation_as_of`, and `top_k`. Defaults are 30/30 days, `new-customer`, `SP`, `2018-08-01`, and 10. Set `recommendation_customer_id` to an actual `customer_unique_id` for personalized candidate features; the default uses the new-customer fallback. The recommendation cutoff should fall within the historical dataset, rather than today's date.

The recommendation feature task builds one customer snapshot. Recommendation baseline evaluation independently builds its historical customer cohorts from the warehouse using July/August 2018 evaluation periods. Both baseline programs log to `data/baselines/mlflow.db`, with separate forecasting and recommendation experiments; CSV outputs go to `data/baselines/forecasting/` and `data/baselines/recommendations/`.

Every worker needs access to the same checkout and data directories. For another deployment location, set `OLIST_PROJECT_ROOT` to the checkout's absolute path and `OLIST_PYTHON` to the worker's absolute Python executable path. By default, the DAG uses the checkout's `.venv/bin/python` if present, otherwise Airflow's Python. The forecasting feature builder processes a large dense dataset in memory; provide enough worker memory for the full Olist run.

This DAG does not reset existing files or run at import time. Stop Airflow before using the reset utilities, since the local development Airflow database and logs are under `data/airflow/`.

## Train production model candidates

Reusable implementations live in `src/models/production/`; CLI wrappers live in `scripts/models/production/`.

Both production modules expose the same entry points. Start with `run_pipeline()`:

```text
prepare_data() → train_model() → evaluate_model() → save_run()
```

Data preparation and evaluation differ by model, but the procedure and names match.
Both files define the same public functions: `predict_baseline()`, `prepare_data()`,
`train_model()`, `evaluate_model()`, `run_pipeline()` and `main()`. The shared
`save_run()` implementation is imported from `common.py`. Model-specific loading
helpers use private names: `_feature_query()` and `_load_labels()`.
Both model wrappers expose `predict()`, and feature preparation uses `prepare_features()`.
`common.save_run()` saves the model and metrics, records them in
MLflow, and updates the active model when the checks pass. Standard operations use
the existing libraries: scikit-learn calculates MAE/MSE, pandas handles category
codes, feature alignment, deduplication and batch aggregation, and MLflow supplies
run IDs and JSON report logging. Custom code handles date boundaries, candidate
rules, fallback selection and model approval.

Both models default to **100 trees** (`--n-estimators`) and **seed 42** (`--seed`).
The estimator constructors specify the task objective, evaluation metric, tree count,
seed and two CPU threads. Advanced tree and regularization settings use LightGBM defaults.
Forecasting can stop early based on validation MAE; recommendations train for the
configured tree count. The same defaults are used by the Airflow DAG.

| Program | Purpose |
|---|---|
| `validate_inputs.py` | Check feature schemas, split metadata, dates and warehouse readability |
| `train_forecasting.py` | Train a LightGBM Poisson regressor for next-day product/state demand |
| `train_recommendation.py` | Train a LightGBM LambdaRank model from historical customer snapshots |

First complete `olist_end_to_end_pipeline`. Disconnect DBeaver from `olist.duckdb` before training, and avoid running warehouse preparation at the same time. Run from the repository root:

```bash
.venv/bin/python scripts/models/production/validate_inputs.py
.venv/bin/python scripts/models/production/train_forecasting.py
.venv/bin/python scripts/models/production/train_recommendation.py
```

The separate Airflow DAG **`olist_production_training`** runs these three commands in order. It appears alongside `olist_end_to_end_pipeline` when Airflow loads `pipelines/`. Unpause and trigger it manually after the data pipeline succeeds. It does not download or rebuild data. The trigger form exposes `max_train_rows`, `forecast_estimators`, `ranking_estimators`, `candidate_limit` and `top_k` (which must not exceed `candidate_limit`). CLI `--help` also exposes historical recommendation cutoff dates and custom artifact/database paths.

### Forecasting evaluation

- Forecasting baselines and production models report only **MAE** (average absolute error) and **RMSE** (larger errors receive more weight). Lower is better.
- The target is `target_demand_next_day`: observations through date `t` predict `t+1`.
- Product/state encodings are fitted on training data; unknown IDs receive a missing-category code.
- Calendar fields describe the forecast date. Neither tomorrow's target nor future observations are model inputs.
- At most 1,000,000 training rows are used by default. Positives and zeros are sampled separately with inverse sampling weights, preserving the population objective while retaining rare positive examples. Validation/test metrics stream over the full evaluated splits.
- Training labels at the validation boundary and validation labels at the test boundary are purged. Early stopping uses a fixed validation sample; model/baseline comparisons use the full remaining validation period.
- Baselines are evaluated for the same next-day target. Their metrics are not interchangeable with the original same-day baseline notebook results. Weekly/28-day seasonal rules use lags 6/27 relative to feature date `t`. Training reads these precomputed columns directly; it does not rebuild features. If older Parquet files lack `demand_lag_6` or `demand_lag_27`, rebuild them with `python scripts/features/build_forecasting_features.py` before production training.

### Recommendation evaluation

- Monthly March–June 2018 snapshots provide training examples; July is validation and August is test by default.
- A snapshot uses only purchases made and delivered before its cutoff. Training delivery outcomes must be known before validation; validation outcomes before test; test outcomes before the test-period end.
- Candidates come from global popularity, recent customer categories and purchase history. Future purchased products are never injected into candidate lists. Repeat purchases are eligible for both ranker and baselines.
- Training omits groups without positive/negative candidate variation because they supply no pairwise ranking signal. Evaluation retains all purchasing customers, including new customers and purchases outside the candidate/catalog pool.
- Recommendation baselines and production models report only **Precision@K** (the fraction of recommendations purchased) and **Recall@K** (the fraction of purchases included in the recommendations). Higher is better. Both model and baselines use the same customers. Warm/new breakdowns remain available in baseline CSV artifacts, using the same two metrics.
- The ranker trains for the configured `ranking_estimators` rounds. Validation selects the baseline by Recall@K, with Precision@K as a tiebreaker; test checks the frozen model. Forecasting still uses early stopping, monitoring MAE.
- Inventory, recalls, views, brands and promotions are not supplied by Olist and are not fabricated.

### Model files and release checks

Each run saves three files in a unique directory under `data/production/forecasting/` or `data/production/recommendations/`:

- `candidate.pkl`: the trained model bundle, including its preprocessing and fallback rule.
- `metrics.csv`: validation and test metrics for the model and all baselines.
- `report.json`: run parameters, evaluation results, approval and publication decisions.

Both programs log the bundle, metrics and report to the existing separate MLflow experiments. The simplified trainers no longer export a second model format, a separate `selected.pkl`, feature schemas/importances, sample predictions or cohort CSVs. Existing run directories are preserved.

The strongest nonzero forecasting baseline and strongest recommendation baseline are selected using validation only. A trained model must improve validation MAE without worse RMSE (forecasting), or improve Recall@K without worse Precision@K (recommendations). Constant forecasting predictions are rejected. The same frozen candidate must also pass those metric comparisons against that baseline on test before publication. Test is a release veto, not a source for selecting hyperparameters or baseline methods.

New runs log only these two performance metrics per task for validation and test. Row/customer counts stay in reports or CSVs, and approval outcomes are MLflow tags. Historical MLflow runs and existing artifacts retain the metrics recorded when they were created.

| Active artifact | Interface |
|---|---|
| `models/forecast_model.pkl` | `joblib.load(path).predict(feature_frame)` |
| `models/recommender.pkl` | `joblib.load(path).predict(snapshot, customer_id, k)` |

A rejected candidate never replaces an existing active file. On the first run, a baseline fallback bundle is written if the candidate fails. Inspect `approved`, `validation_approved`, `holdout_passed` and `publication` in the run report; successful training/DAG execution does not mean the learned model was approved. Load `.pkl` bundles only from trusted runs in this repository. They include the estimator and preprocessing/fallback configuration and can be loaded with `joblib.load()`.

To use the saved recommendation bundle:

```python
import joblib
import pandas as pd
from src.models.baseline.recommendation_baselines import load_history
from src.models.production.recommendation import RecommendationSnapshot

cutoff = pd.Timestamp("2018-08-01")
history = load_history(warehouse_path, cutoff)  # warehouse_path is a pathlib.Path
snapshot = RecommendationSnapshot(history, cutoff, candidate_limit=100)
model = joblib.load("models/recommender.pkl")
products = model.predict(snapshot, "CUSTOMER_UNIQUE_ID", k=10)
```

The forecasting frame must contain observed demand, historical lag/rolling columns, date, product and state. Weekly/28-day fallback rules additionally require `demand_lag_6`/`demand_lag_27`; they do not require a future target column. These scripts provide training, backtesting and local inference artifacts; live feature generation, an API deployment and live inventory eligibility are separate work.

## View MLflow runs in the browser

After running the baseline programs or the Airflow pipeline, open a new terminal in the repository root and start MLflow with the same tracking database used by the pipeline:

```bash
.venv/bin/mlflow server \
  --backend-store-uri sqlite:///data/baselines/mlflow.db \
  --host 127.0.0.1 \
  --port 5001
```

Open [http://localhost:5001](http://localhost:5001) and select an experiment:

- **`ecommerce-demand-forecasting`**: forecasting baseline runs.
- **`ecommerce-recommendations`**: recommendation baseline runs.

Click a run to inspect its parameters, validation/test metrics, tags and saved CSV artifacts. Select multiple runs and click **Compare** to compare their performance.

Keep the terminal running while using the UI; press **Ctrl+C** to stop the MLflow server. Airflow remains available separately at [http://localhost:8080](http://localhost:8080). If port 5001 is already occupied, choose another port and use that port in the browser URL.

If the experiment list is empty, confirm that baseline evaluation has finished and that the server was started from the repository root with `data/baselines/mlflow.db` as its backend store. Starting MLflow against another database will show different runs.

## Reset local pipeline outputs

To remove only local data and databases, use:

```bash
.venv/bin/python scripts/maintenance/cleanup_data.py --dry-run
.venv/bin/python scripts/maintenance/cleanup_data.py --yes
```

This deletes the entire root `data/` directory, including downloads, processed/curated tables, features, DuckDB/SQLite warehouses, and `data/baselines/mlflow.db`. It leaves notebook outputs, model files and MLflow artifact directories outside `data/` in place; the deleted MLflow database's run records are not retained. Custom paths outside `data/` and remote storage are not removed. Stop processes using these files before applying cleanup.

Preview the reset (nothing is deleted):

```bash
.venv/bin/python scripts/maintenance/reset_repository.py --dry-run
```

Apply the reset:

```bash
.venv/bin/python scripts/maintenance/reset_repository.py --yes
```

This permanently removes the repository's root `data/` (including raw downloads, processed/curated data, features, warehouses and the MLflow database), `mlruns/`, `mlartifacts/`, `artifacts/`, and `models/`. It also removes `__pycache__` directories and loose `.pyc`/`.pyo` files throughout project folders, other local code caches, and saved notebook outputs and execution counts. Source code, notebook code and explanations, configuration, documents, Git history, credentials, and virtual environments are retained; their internal caches are excluded. Running without arguments is a dry run.

Stop running pipelines, notebooks and the MLflow server before applying a reset. Custom output locations outside the listed folders and remote storage are not reset. Afterwards, rerun the data download, validation, cleaning and curation commands above, then rebuild features and evaluate baselines.
