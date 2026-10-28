# ecommerce-forecasting-recommender
Production-ready MLOps platform for e-commerce demand forecasting and personalised product recommendations using the Olist dataset, with DVC versioning, MLflow experiment tracking, Airflow orchestration, FastAPI serving, CI/CD, and drift monitoring.

Explore [the simple baselines notebook](notebooks/simple_baselines.ipynb) for worked examples, forecasting and recommendation evaluation, and local MLflow experiment logging. Run the data preparation and forecasting feature scripts first, then execute the notebook from top to bottom using the project environment with `ipykernel` installed. Results and the MLflow database are written to `data/baselines/`.

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
└── models/
    └── baseline/
        ├── forecasting_baselines.py
        └── recommendation_baselines.py
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
