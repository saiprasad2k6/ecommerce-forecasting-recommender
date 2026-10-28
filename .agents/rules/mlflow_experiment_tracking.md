# MLflow Experiment Tracking & Model Evaluation Conventions

When building forecasting or recommendation models (baselines, tree-based models, neural models) in this repository, adhere to the following MLflow experiment tracking conventions:

## 1. Dedicated Experiments
- Keep forecasting and recommendation experiments strictly separated:
  - Forecasting: `ecommerce-demand-forecasting`
  - Recommendations: `ecommerce-recommendations`
- Store tracking databases locally in `data/baselines/mlflow.db` or configured remote URI.

## 2. Champion Selection Logic (Sparsity Guardrail)
- When comparing models on sparse target data (e.g., daily item sales):
  - Always filter out trivial diagnostic models (such as `always_zero`):
    ```python
    non_trivial = val_metrics[val_metrics["baseline"] != "always_zero"]
    champion_baseline = non_trivial.sort_values("mae").iloc[0]["baseline"]
    ```
  - Base the champion selection strictly on **validation split** metrics (e.g. `val_mae` or `val_wape`), never on the test split.

## 3. Standardized MLflow Logging Pattern
For every model/baseline evaluated, structure the MLflow run as follows:

### A. Parameters
- `task`: `"demand_forecasting"` or `"recommendation"`
- `model_family`: `"heuristic_baseline"`, `"gradient_boosted_trees"`, etc.
- `baseline_name` / `model_name`: Descriptive name (e.g., `"same_day_last_week"`, `"lightgbm_regressor"`)
- `window_days`: Trailing window size if applicable (e.g., 7, 28)
- `pipeline_feature_store`: Path to upstream feature parquets (e.g., `"data/features/forecasting"`)
- `validation_range` & `test_range`: Extracted from feature store `metadata.json`
- `is_champion_baseline`: Boolean indicating if selected on validation

### B. Tags
- `pipeline_integrated`: `"true"`
- `selected_on_validation`: `"true"` | `"false"`
- `model_status`: `"champion_baseline"` (only on winning run)

### C. Dual-Split Metrics
Always log metrics for both splits with explicit prefixes:
- `val_mae`, `val_rmse`, `val_wape`, `val_positive_mae`, `val_zero_fraction`
- `test_mae`, `test_rmse`, `test_wape`, `test_positive_mae`, `test_zero_fraction`

### D. Artifacts
- Save and log evaluation metric tables (`metrics_{baseline}.csv`, `forecasting_metrics_all.csv`).
- For champion runs, log sample active series prediction comparisons (`sample_active_forecasts.csv`).
