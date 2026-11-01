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
│   ├── forecasting/
│   │   ├── forecasting_baseline.py
│   │   └── forecasting_lightgbm.py
│   ├── recommendation/
│   │   ├── recommendation_baseline.py
│   │   └── recommendation_lightgbm.py
│   ├── comparison/
│   │   └── compare_models.py
│   └── validate_inputs.py
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
.venv/bin/python scripts/models/forecasting/forecasting_baseline.py
```

For recommendation features, supply a customer, state and historical cutoff:

```bash
.venv/bin/python scripts/features/build_recommendation_features.py \
  --customer-id CUSTOMER_UNIQUE_ID --state SP --as-of 2018-08-01
.venv/bin/python scripts/models/recommendation/recommendation_baseline.py --help
```

Use `--help` on any script to inspect its arguments. Import reusable functions from `src.data`, `src.features`, `src.models.forecasting` and `src.models.recommendation` in application code and notebooks.

## Prepare data and features with Airflow

The DAG in `pipelines/data_pipeline.py` is named `olist_data_preparation` and runs:

```text
download_olist_data
  → validate_olist_data
  → clean_olist_data
  → prepare_curated_data
  → build_forecasting_features
  → build_recommendation_features
```

It is manually triggered, with one active run and one task at a time. Later tasks run only after their dependencies succeed. Complete raw downloads are reused. After it succeeds, trigger `olist_model_training` to evaluate baselines, train candidates, compare results and publish the selected models.

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

Open the address printed by Airflow `http://localhost:8080.`, unpause `olist_data_preparation` and trigger it. In another terminal with the same environment variables, you can also trigger it with:

```bash
.venv/bin/airflow dags trigger olist_data_preparation
```

Shared defaults live in `configs/settings.yaml`. The trigger form exposes
`validation_start=2018-06-01`, `test_start=2018-07-01`,
`monitoring_start=2018-08-01` and `monitoring_end=2018-09-01` (exclusive),
plus the customer snapshot parameters (`new-customer`, `SP`, `2018-08-01`, K=10).
Forecasting train/validation/test splits use **target dates**, not feature dates:
May 31 features predict June 1 and belong to validation. August target outcomes
are excluded from development splits. A separate `inference.parquet` contains
August feature dates and does not require next-day labels.

Keep replay boundaries identical in data preparation and training when overriding
DAG parameters. Recommendation model jobs receive the same June validation, July
test and August release boundaries. Forecasting training reads its boundaries from
the generated feature metadata. Change the shared configuration before starting a
new replay and rebuild the data rather than mixing outputs from different windows.

The recommendation feature task builds one customer snapshot. In the separate `olist_model_training` DAG, recommendation baseline evaluation builds its historical customer cohorts from the warehouse using June/July 2018 evaluation periods. Both baseline programs log to `data/baselines/mlflow.db`, with separate forecasting and recommendation experiments; CSV outputs go to `data/baselines/forecasting/` and `data/baselines/recommendations/`.

Every worker needs access to the same checkout and data directories. For another deployment location, set `OLIST_PROJECT_ROOT` to the checkout's absolute path and `OLIST_PYTHON` to the worker's absolute Python executable path. By default, the DAG uses the checkout's `.venv/bin/python` if present, otherwise Airflow's Python. The forecasting feature builder processes a large dense dataset in memory; provide enough worker memory for the full Olist run.

This DAG does not reset existing files or run at import time. Stop Airflow before using the reset utilities, since the local development Airflow database and logs are under `data/airflow/`.

## Train production model candidates

Models are grouped by task: forecasting and recommendation. Filenames identify
the task and algorithm, such as `forecasting_baseline.py`, `forecasting_lightgbm.py`
and `recommendation_lightgbm.py`. Source implementations and CLI wrappers use the
same layout under `src/models/` and `scripts/models/`. LightGBM is the currently
implemented trained algorithm; additional models can be registered as separate jobs.

Shared model support lives directly under `src/models/`:

```text
src/models/
├── common.py
├── datasets.py
├── evaluation.py
├── comparison.py
├── forecasting/
│   ├── forecasting_baseline.py
│   └── forecasting_lightgbm.py
└── recommendation/
    ├── recommendation_baseline.py
    └── recommendation_lightgbm.py
```

Both LightGBM modules follow the same procedure:

```text
prepare_data() → train_model() → evaluate_model() → save_run()
```

All four modules expose `run_pipeline()`, `main()`, a model `predict()` method,
and `evaluate_model` imported from the shared task evaluator. Evaluation always
receives a dictionary of named models. Only LightGBM modules need `prepare_data()`
and `train_model()`; baselines simply apply their prediction rules. The baseline
modules also expose `predict_baseline()`. Recommendation pipelines use the same
`warehouse` and `top_k` argument names. CLI output identifies the pipeline run;
detailed metrics and model artifacts remain in MLflow.

Baseline evaluation and trained models run independently from their task folders. Both use
`src/models/datasets.py` for input validation, shared data loading and recommendation features, and
`src/models/evaluation.py` for Precision/Recall and model evaluation.
The validation CLI remains `scripts/models/validate_inputs.py`; its checks live in `datasets.py`. Each job logs its own
model and scores; training does not execute baselines or publish active models.
`src/models/comparison.py` reads the completed MLflow runs and makes the release decision.
Its `compare_models()` function coordinates three steps: `load_model_results()` loads
the completed runs, `compare_model_results()` selects and checks the candidate, and
`save_comparison()` records the decision and publishes the selected model.

Both models default to **100 trees** (`--n-estimators`) and **seed 42** (`--seed`).
The estimator constructors specify the task objective, evaluation metric, tree count,
seed and two CPU threads. Advanced tree and regularization settings use LightGBM defaults.
Both models can stop early after 20 rounds without validation improvement:
forecasting monitors MAE, and recommendations monitor Precision@K.
The tree count is an upper limit. The same defaults are used by the Airflow DAG.

| Program | Purpose |
|---|---|
| `validate_inputs.py` | Check feature schemas, split metadata, dates and warehouse readability |
| `forecasting_lightgbm.py` | Train a LightGBM Poisson regressor for next-day product/state demand |
| `recommendation_lightgbm.py` | Train a LightGBM LambdaRank model from historical customer snapshots |

First complete `olist_data_preparation`. Disconnect DBeaver from `olist.duckdb` before training, and avoid running warehouse preparation at the same time. Run from the repository root:

With the current model configuration, **`olist_model_training`** runs six tasks:

```text
validate_training_inputs
  → evaluate_forecasting_baselines
  → evaluate_recommendation_baselines
  → train_forecasting_lightgbm
  → train_recommendation_lightgbm_ranker
  → compare_models
```

Trigger it from the Airflow UI after `olist_data_preparation` succeeds, or use:

```bash
airflow dags trigger olist_model_training
```

The DAG appears when Airflow loads `pipelines/`. It uses its Airflow run ID as the
`pipeline_run_id` for all jobs. Tasks run one at a time for the local SQLite/DuckDB
setup. The trigger form exposes training row limits, tree counts, candidate limits
and Top K. Input data is reused; this DAG does not download or rebuild it.

To run the same workflow manually, use a **new pipeline run ID for each workflow**
and pass the same ID to every job:

```bash
PIPELINE_RUN_ID="manual-$(date +%Y%m%d-%H%M%S)"
.venv/bin/python scripts/models/validate_inputs.py
.venv/bin/python scripts/models/forecasting/forecasting_baseline.py --pipeline-run-id "$PIPELINE_RUN_ID"
.venv/bin/python scripts/models/recommendation/recommendation_baseline.py --pipeline-run-id "$PIPELINE_RUN_ID"
.venv/bin/python scripts/models/forecasting/forecasting_lightgbm.py --pipeline-run-id "$PIPELINE_RUN_ID"
.venv/bin/python scripts/models/recommendation/recommendation_lightgbm.py --pipeline-run-id "$PIPELINE_RUN_ID"
.venv/bin/python scripts/models/comparison/compare_models.py --pipeline-run-id "$PIPELINE_RUN_ID"
```

Standalone jobs can also run without an ID for exploration; each generates a new
ID and therefore will not automatically form a comparison group. When changing
recommendation dates, `--top-k` or `--candidate-limit`, pass identical evaluation
settings to both recommendation jobs.

### Add another model

`configs/production_models.json` is shared by the training DAG and comparison code.
Each key under `forecasting` or `recommendation` is the model name expected in the
MLflow `model` tag and saved result. Each entry specifies a script relative to
`scripts/models/` and maps CLI flags to existing Airflow parameters.

For example, after implementing a random-forest forecaster, add this entry under
`forecasting` alongside `lightgbm`:

```json
"random_forest": {
  "script": "forecasting/forecasting_random_forest.py",
  "parameters": {
    "max-train-rows": "max_train_rows"
  }
}
```

1. Add the implementation as `src/models/<task>/<task>_<algorithm>.py` and its
   CLI wrapper as `scripts/models/<task>/<task>_<algorithm>.py`. For example,
   use `forecasting/forecasting_model2.py` for another forecasting model. Keep familiar stage
   names such as `prepare_data()`, `train_model()`, `evaluate_model()` and `run_pipeline()`.
2. Reuse shared datasets and evaluation functions. Forecasting models must expose
   `predict(feature_frame)`; recommenders must expose `predict(snapshot, customer, k)`.
3. Log through `common.save_run()` with the matching `task`, `role="candidate"`,
   registered model name, pipeline run ID, evaluation context, validation/test
   scores and a loadable `model.pkl`. Forecasting results also need `prediction_varies`.
4. Add the model to the configuration. Its wrapper must accept `--pipeline-run-id`
   and the flags in its `parameters` mapping. For new tuning options, add the
   corresponding Airflow `Param`; an algorithm need not accept LightGBM's flags.
5. Let Airflow reload the DAG, then trigger a new pipeline run. The DAG creates
   `train_<task>_<model>` automatically and comparison waits for every configured
   model. To run manually, execute every registered job with the same pipeline ID
   before the comparison command.

A model can be registered for forecasting, recommendations or both. Adding a
registration does not implement its trainer. Keep configuration fixed during a
pipeline execution; newly registered models must produce matching results before
comparison can proceed.

All model implementations and CLI wrappers live in the forecasting and
recommendation folders. There are no baseline/production model directories.
Saved models use the current module paths and load directly with `joblib.load()`.

### Forecasting evaluation

- Forecasting baselines and production models report only **MAE** (average absolute error) and **RMSE** (larger errors receive more weight). Lower is better.
- The target is `target_demand_next_day`: observations through date `t` predict `t+1`.
- Product/state encodings are fitted on training data; unknown IDs receive a missing-category code.
- Calendar fields describe the forecast date. Neither tomorrow's target nor future observations are model inputs.
- At most 1,000,000 training rows are used by default. Positives and zeros are sampled separately with inverse sampling weights, preserving the population objective while retaining rare positive examples. Validation/test metrics stream over the full evaluated splits.
- Training labels at the validation boundary and validation labels at the test boundary are purged. Early stopping uses a fixed validation sample; model/baseline comparisons use the full remaining validation period.
- Baselines are evaluated for the same next-day target. Their metrics are not interchangeable with the original same-day baseline notebook results. Weekly/28-day seasonal rules use lags 6/27 relative to feature date `t`. Training reads these precomputed columns directly; it does not rebuild features. If older Parquet files lack `demand_lag_6` or `demand_lag_27`, rebuild them with `python scripts/features/build_forecasting_features.py` before production training.

### Recommendation evaluation

- Monthly March–May 2018 snapshots provide training examples; June is validation, July is test and August is reserved for inference/monitoring by default.
- A snapshot uses only purchases made and delivered before its cutoff. Training delivery outcomes must be known before validation; validation outcomes before test; test outcomes before the test-period end.
- Candidates come from global popularity, recent customer categories and purchase history. Future purchased products are never injected into candidate lists. Repeat purchases are eligible for both ranker and baselines.
- Training omits groups without positive/negative candidate variation because they supply no pairwise ranking signal. Evaluation retains all purchasing customers, including new customers and purchases outside the candidate/catalog pool.
- Recommendation baselines and production models report only **Precision@K** (the fraction of recommendations purchased) and **Recall@K** (the fraction of purchases included in the recommendations). Higher is better. Both model and baselines use the same customers and delivery-outcome boundaries.
- The ranker uses monthly validation ranking groups for early stopping, with historical features built before each snapshot and delivery outcomes known before the test cutoff. It monitors Precision@K and stops after 20 rounds without improvement, up to `ranking_estimators` rounds. Training and early-stopping groups require both positive and negative candidates; final evaluation still includes all purchasing customers.
- Precision@K is supplied to LightGBM through a small evaluation callback because it is not a built-in ranking metric. MLflow records Precision@K and Recall@K, plus the training start, random seed, estimator limit, K and candidate limit. Early stopping waits 20 rounds without validation improvement. Validation selects the baseline by Recall@K, with Precision@K as a tiebreaker; test checks the frozen model and is not passed to `fit()`.
- Inventory, recalls, views, brands and promotions are not supplied by Olist and are not fabricated.

### Model files and release checks

Each baseline/candidate run saves three files in its own MLflow run directory:

- `model.pkl`: that run's model, with its preprocessing or baseline rule.
- `metrics.csv`: that model's validation and test metrics.
- `report.json`: evaluation settings, data fingerprint, run ID and parameters.

Baseline artifacts are under `data/baselines/forecasting/` and
`data/baselines/recommendations/`; LightGBM candidates are under
`data/production/lightgbm/forecasting/` and `data/production/lightgbm/recommendations/`. Historical runs remain unchanged.

Only the final comparison task updates active model files. It requires all baseline
runs and every configured candidate from the same pipeline run ID, with matching evaluation
fingerprints, settings and row/customer counts. Missing or incompatible runs fail
the task before either model is published. Task retries use the newest completed
run for each method within that group; failed runs and other groups are excluded.
Comparison decisions are logged as separate MLflow runs and saved under
`data/comparisons/<comparison-run-id>/report.json`.

Validation selects the strongest nonzero forecasting baseline and strongest recommendation baseline. A candidate qualifies by improving validation MAE without worse RMSE (forecasting), or improving Recall@K without worse Precision@K (recommendations). Constant forecasting candidates are excluded. Among qualifying candidates, forecasting selects the lowest validation MAE, then RMSE; recommendations select the highest validation Recall@K, then Precision@K. Model names break remaining ties deterministically.

Only the selected validation winner is checked against its baseline on test for publication. If it fails, comparison retains the current model or publishes a baseline fallback on the first deployment. It does not try other candidates based on test results. If no candidate qualifies on validation, none is approved. Test metrics never determine which candidate wins. Comparison reports include `validation_ranking`, `validation_eligible` and `candidate_run_ids` for all configured candidates.

New runs log only these two performance metrics per task for validation and test. Row/customer counts stay in reports or CSVs, and approval outcomes are MLflow tags. Historical MLflow runs and existing artifacts retain the metrics recorded when they were created.

| Active artifact | Interface |
|---|---|
| `models/forecast_model.pkl` | `joblib.load(path).predict(feature_frame)` |
| `models/recommender.pkl` | `joblib.load(path).predict(snapshot, customer_id, k)` |

A rejected candidate never replaces an existing active file. On the first run, a baseline fallback bundle is written if the candidate fails. Inspect `approved`, `validation_approved`, `holdout_passed` and `publication` in the comparison report; successful training/DAG execution does not mean the learned model was approved. Load `.pkl` bundles only from trusted runs in this repository. They contain the estimator and preprocessing, or the selected baseline rule and can be loaded with `joblib.load()`.

To use the saved recommendation bundle:

```python
import pandas as pd
import joblib
from src.models.datasets import load_history
from src.models.datasets import RecommendationSnapshot

cutoff = pd.Timestamp("2018-08-01")
history = load_history(warehouse_path, cutoff)  # warehouse_path is a pathlib.Path
snapshot = RecommendationSnapshot(history, cutoff, candidate_limit=100)
model = joblib.load("models/recommender.pkl")
products = model.predict(snapshot, "CUSTOMER_UNIQUE_ID", k=10)
```

The forecasting frame must contain observed demand, historical lag/rolling columns, date, product and state. Weekly/28-day baseline rules additionally require `demand_lag_6`/`demand_lag_27`; they do not require a future target column. These scripts provide training, backtesting and local inference artifacts; live feature generation, an API deployment and live inventory eligibility are separate work.

## Batch inference from the selected model

Inference loads only the active bundle published by comparison:

- Forecasting: `models/forecast_model.pkl`.
- Recommendations: `models/recommender.pkl`.

It calls the bundle's `predict()` interface without choosing an algorithm, scanning
MLflow runs or importing a specific LightGBM trainer. The active bundle can contain
a trained model or the selected baseline fallback. A rejected candidate cannot be
used just because it is the newest training run.

Comparison stores `serving_metadata` inside each newly published bundle: task,
model name, candidate/baseline role, selected model run ID, originating training
pipeline run ID, serving approval and recommendation K/candidate settings. Model
and metadata are replaced atomically together. Retaining an existing active model
also retains its original identity. Inference refuses missing bundles or bundles
without matching serving approval.

The **`olist_batch_inference`** DAG runs `predict_forecasting` then
`predict_recommendation`. Trigger it after model comparison. Its form accepts
`as_of`, `forecast_features`, `warehouse`, `models_dir` and `output_dir`. Default
inputs replay August 1, 2018 from `inference.parquet`. This is a historical replay;
for real batches, provide inputs prepared using information available at the cutoff.

Monitoring defaults are also recorded in `configs/settings.yaml`: a seven-day
window, a seven-day recommendation outcome horizon, MAE/RMSE and Precision@K/Recall@K.
Automatic retraining is disabled. The monitoring job is not implemented yet.

Equivalent manual commands:

```bash
.venv/bin/python scripts/inference/run_batch.py \
  --task forecasting --as-of 2018-08-01 \
  --input-path data/features/forecasting/inference.parquet

.venv/bin/python scripts/inference/run_batch.py \
  --task recommendation --as-of 2018-08-01 \
  --input-path data/warehouse/olist.duckdb
```

Forecasting filters the input Parquet to feature date `as_of` and predicts the next
day. It requires the selected model's input columns but does not require
`target_demand_next_day`. Recommendations use purchases made and delivered strictly
before `as_of` and recommend for known historical customers using the selected
model's saved K and candidate limit. The two tasks load their own active model at
task start; a later model replacement takes effect on a subsequent task execution.

Outputs are under
`data/predictions/<task>/as_of=<date>/model_run_id=<selected-run>/predictions.parquet`.
They contain model identity and both training/inference pipeline IDs. Recommendation
lists preserve rank. Retries for the same model/date replace the output atomically;
different selected model run IDs keep separate prediction partitions.


## View MLflow runs in the browser

After running the baseline programs or the Airflow pipeline, open a new terminal in the repository root and start MLflow with the same tracking database used by the pipeline:

```bash
.venv/bin/mlflow server \
  --backend-store-uri sqlite:///data/baselines/mlflow.db \
  --host 127.0.0.1 \
  --port 5001
```

Open [http://localhost:5001](http://localhost:5001) and select an experiment:

- **`ecommerce-demand-forecasting`**: forecasting baseline, candidate and comparison runs.
- **`ecommerce-recommendations`**: recommendation baseline, candidate and comparison runs.

Click a run to inspect its parameters, validation/test metrics, tags and saved CSV artifacts. Filter by the `pipeline_run_id` tag, select baseline/candidate runs and click **Compare**. The `role` tag identifies `baseline`, `candidate` and `comparison` runs. Baseline and candidate runs use the same metric names (`val_mae`, `test_mae`, or `val_recall_at_k`, for example).

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
