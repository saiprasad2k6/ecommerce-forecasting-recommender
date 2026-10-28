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
│   └── baseline/
│       ├── forecasting_baselines.py
│       └── recommendation_baselines.py
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
