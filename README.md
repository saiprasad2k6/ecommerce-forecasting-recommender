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
