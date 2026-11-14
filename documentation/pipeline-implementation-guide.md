# Step-by-Step MLOps Implementation Guide

This guide describes how to build the demand forecasting and recommendation solution from the assignment brief. It is designed as a practical implementation plan that can begin locally and later move to cloud infrastructure.

## 1. Define the Scope and Success Criteria

Build two related ML capabilities:

1. **Demand forecasting:** Predict product-level demand for the next day or next week.
2. **Recommendations:** Rank products that are relevant to each customer and their current context.

Start with a narrow first release. For example:

- Forecast daily demand for the top 1,000 products.
- Recommend the top 10 products for active customers.
- Support a fallback recommendation list for anonymous or new customers.

Define business KPIs before training any model:

| Area | KPI | Purpose |
| --- | --- | --- |
| Recommendations | Click-through rate, conversion rate, revenue per session | Measure recommendation quality and business impact |
| Recommendations | Add-to-cart rate and recommendation coverage | Measure usefulness and how many users receive recommendations |
| Forecasting | MAE, RMSE, weighted MAPE | Measure forecast accuracy |
| Inventory | Stockout rate and excess inventory rate | Measure operational impact |
| Platform | API latency, error rate, throughput | Ensure the service remains reliable during traffic spikes |
| MLOps | Drift score, model age, retraining success rate | Measure production health |

Set explicit acceptance thresholds, such as:

- Forecast weighted MAPE below the agreed business threshold.
- Recommendation conversion rate better than the existing baseline.
- Prediction API p95 latency below 200 ms.
- No deployment unless validation, performance, and data-quality checks pass.

## 2. Create the Project Structure

Create a repository with separate modules for data, features, models, pipelines, serving, and tests:

```text
mlops-retail-platform/
├── data/
│   ├── raw/
│   ├── processed/
│   └── reference/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── monitoring/
│   └── common/
├── pipelines/
│   ├── data_pipeline.py
│   ├── training_pipeline.py
│   ├── batch_inference.py
│   └── monitoring_pipeline.py
├── api/
│   └── app.py
├── tests/
├── configs/
│   ├── development.yaml
│   └── production.yaml
├── artifacts/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

Keep configuration, secrets, and code separate. Store credentials in environment variables or a secret manager rather than in the repository.

## 3. Set Up the Development Environment

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the initial dependencies:

```bash
pip install pandas numpy scikit-learn lightgbm xgboost
pip install mlflow dvc evidently pandera
pip install fastapi uvicorn pydantic
pip install apache-airflow
pip install pytest docker
```

Freeze the working environment after testing:

```bash
pip freeze > requirements.txt
```

Use Docker Compose for local services such as PostgreSQL, MLflow, Airflow, and an object store. Use managed equivalents in production when appropriate.

### 3.1 Download the Olist Dataset

Use the **Brazilian E-Commerce Public Dataset by Olist** as the single dataset for this project. It contains approximately 100,000 orders from 2016 to 2018 and is suitable for purchase-history recommendations and order-based demand forecasting.

Download it from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). You can download the ZIP file through the browser, or use the Kaggle CLI after configuring your Kaggle API credentials:

```bash
mkdir -p data/raw
pip install kaggle
kaggle datasets download -d olistbr/brazilian-ecommerce -p data/raw --unzip
```

The raw directory should contain these files:

```text
data/raw/
├── olist_customers_dataset.csv
├── olist_geolocation_dataset.csv
├── olist_order_items_dataset.csv
├── olist_order_payments_dataset.csv
├── olist_order_reviews_dataset.csv
├── olist_orders_dataset.csv
├── olist_products_dataset.csv
├── olist_sellers_dataset.csv
└── product_category_name_translation.csv
```

Do not commit the CSV files to Git. Track them with DVC in the next step. Review the dataset terms on Kaggle before using the data outside the assignment.

This dataset does not contain product views, clicks, promotions, or inventory snapshots. Use purchase history for the first recommendation model and order history for demand forecasting. Add those missing sources later if this becomes a real production system.

## 4. Design the Data Contracts

Define the minimum input tables before writing the pipeline.

### Customer and Product Data

Include stable identifiers and descriptive attributes:

Use these Olist files:

- `olist_customers_dataset.csv`: `customer_id`, `customer_unique_id`, `customer_zip_code_prefix`, `customer_city`, and `customer_state`
- `olist_products_dataset.csv`: `product_id`, product category, dimensions, weight, and photos
- `olist_sellers_dataset.csv`: `seller_id`, seller location, and seller region
- `product_category_name_translation.csv`: Portuguese-to-English category names

The Olist dataset does not contain customer consent, brand, size, colour, or promotion fields. Do not invent these fields. Add them only when a future source system supplies them.

### Interaction Data

Use completed purchases as the initial customer-product interaction:

- `olist_order_items_dataset.csv`: `order_id`, `order_item_id`, `product_id`, `seller_id`, `shipping_limit_date`, `price`, and `freight_value`
- `olist_orders_dataset.csv`: order status and purchase, approval, delivery, and estimated-delivery timestamps
- `olist_order_reviews_dataset.csv`: review score and review comments

Join orders to customers through `order_id`, and join order items to products through `product_id`. Treat each delivered order item as a positive purchase interaction. The dataset has no anonymous-session, impression, view, or click events.

### Transaction and Inventory Data

Use these Olist files for transaction features:

- `olist_orders_dataset.csv`: purchase timestamp and order status
- `olist_order_items_dataset.csv`: item price and freight value
- `olist_order_payments_dataset.csv`: payment type, installments, and payment value

Olist does not provide inventory levels, stockout indicators, quantity ordered, or warehouse IDs. For the first version, forecast order-item counts and use seller or customer state as a regional proxy. Add inventory data before making replenishment decisions.

Create schemas with Pandera or Great Expectations. Validate required columns, types, ranges, duplicate keys, null limits, and timestamp freshness.

## 5. Build the Data Ingestion Pipeline

Implement the first Airflow DAG with these tasks:

1. Download or extract the Olist source files.
2. Store an immutable copy in object storage under a dataset-versioned path.
3. Validate every CSV against the Olist data contract.
4. Remove duplicate order items and quarantine invalid records.
5. Standardise timestamps, category names, identifiers, and numeric values.
6. Join orders, customers, products, sellers, payments, and reviews into curated tables.
7. Write cleaned data to a warehouse or feature-store input table.
8. Publish data-quality metrics and a pipeline status.

Use `order_purchase_timestamp` to create a partition such as `purchase_date=YYYY-MM-DD`, so that a failed day can be reprocessed without rewriting all historical data.

Example DAG flow:

```text
extract -> raw_storage -> validate -> clean -> feature_tables -> quality_report
```

Make each task idempotent. Running the same date twice should produce the same result rather than duplicate records.

## 6. Create Forecasting Features

Build features using only information available before the forecast time. Useful features include:

- demand lags: 1, 7, 14, and 28 days
- rolling mean and rolling standard deviation
- day of week, month, season, and holiday indicators
- price, discount, and promotion indicators
- category, brand, and region
- inventory availability and stockout history
- trend and recent growth rate

Prevent leakage by ensuring that future sales and future inventory values are not used to construct historical features.

Create a time-based split:

- training period: oldest historical records
- validation period: following time window
- test period: most recent historical window

Do not use a random split for forecasting when it would allow future information into the training set.

## 7. Create Recommendation Features

Begin with a simple candidate-generation and ranking design.

### Candidate Generation

Generate candidates from:

- most popular products by region and category
- products viewed or purchased by the customer
- products similar to recently viewed products
- products frequently bought together
- currently available promotional products

Remove unavailable, recalled, or already purchased products where appropriate.

### Ranking Features

For each customer-product pair, calculate:

- recent views and purchases
- time since the last interaction
- category and brand affinity
- product popularity
- product price and discount
- inventory availability
- customer segment
- similarity to recent interactions

Train a ranking model or begin with a weighted scoring function as the baseline. Keep the baseline permanently available for fallback and comparison.

## 8. Train Baseline Models First

Build simple baselines before using complex algorithms.

### Forecasting Baselines

- last-day demand
- same-day-last-week demand
- moving average
- seasonal naive forecast

### Recommendation Baselines

- global popularity
- category popularity
- customer purchase history
- popularity filtered by inventory

Record baseline metrics in MLflow. A production model must beat the relevant baseline, not only achieve a good offline score.

## 9. Track Experiments with MLflow

For every training run, log:

- dataset version or DVC hash
- feature version
- model type and hyperparameters
- training and validation time ranges
- evaluation metrics
- model artifact
- environment and source-code version

Use separate MLflow experiments for forecasting and recommendations. Register only models that pass the validation gates.

Recommended model lifecycle:

```text
candidate -> validated -> staging -> production -> archived
```

Never replace the production model without retaining the previous version and its metadata.

## 10. Version Data and Pipelines with DVC

Track large datasets and generated training inputs with DVC:

```bash
dvc init
dvc add data/raw
git add data/raw.dvc .dvc/config
dvc remote add -d local_remote ./dvc-storage
dvc push
```

For a shared project, replace `./dvc-storage` with an S3, Azure Blob, or Google Cloud Storage remote. Keep the DVC pointer file in Git and keep the CSV contents in the DVC remote.

After cleaning and feature generation, version the reproducible outputs as separate artifacts:

```bash
dvc add data/processed data/reference
git add data/processed.dvc data/reference.dvc
git commit -m "Version Olist processed data"
dvc push
```

Create a DVC pipeline with stages such as:

```text
ingest -> validate -> clean_olist -> features -> train -> evaluate
```

Commit the DVC metadata and code to Git. Store the actual dataset in object storage. This makes it possible to reproduce a model from a specific code commit and data version.

## 11. Add Training and Promotion Gates

The training pipeline should perform the following steps:

1. Load a specific approved dataset version.
2. Validate the dataset.
3. Build features.
4. Train the candidate model.
5. Evaluate against the holdout set and baseline.
6. Run unit and integration tests.
7. Check for fairness, feature leakage, and unacceptable performance by segment.
8. Log the run to MLflow.
9. Register the model only if all gates pass.
10. Deploy to staging for a smoke test.
11. Promote to production after approval or an automated policy check.

Do not promote a model because of one metric alone. Combine technical metrics, business metrics, data-quality checks, and service checks.

## 12. Package the Inference Service

Create a FastAPI service with endpoints such as:

```text
GET  /health
POST /forecast
POST /recommendations
```

The service should:

1. Load the approved model and feature transformers at startup.
2. Validate incoming requests with Pydantic.
3. Generate features using the same logic used during training.
4. Return predictions, scores, model version, and request metadata.
5. Log latency, errors, input schema versions, and prediction counts.
6. Return a controlled fallback when the model or feature store is unavailable.

Example response metadata:

```json
{
  "model_version": "recommendation-model-12",
  "generated_at": "2026-10-21T12:00:00Z",
  "recommendations": [
    {"product_id": "P1001", "score": 0.91}
  ]
}
```

## 13. Build Batch Inference

Use batch inference for workloads that do not require a synchronous response:

- daily demand forecasts
- customer email recommendations
- warehouse replenishment forecasts
- product-level promotion suggestions

The batch pipeline should write predictions with a prediction timestamp, model version, feature version, and input data version. Keep the outputs available for auditing and business reporting.

## 14. Containerise and Deploy

Create separate Docker images for training jobs and the inference API. The image should:

- pin dependencies
- run as a non-root user
- expose only the required port
- include a health check
- avoid embedding credentials

Deploy the API behind a load balancer or API gateway. Configure horizontal scaling for seasonal events. Use a queue or asynchronous worker for long-running batch predictions.

Use a staged deployment strategy:

1. Deploy to a development environment.
2. Run smoke and integration tests.
3. Deploy to staging.
4. Send a small percentage of traffic to the new model.
5. Compare health and business metrics.
6. Promote gradually or roll back.

## 15. Add CI/CD Automation

Configure GitHub Actions or another CI/CD system to run on pull requests and model changes.

### Pull Request Checks

- formatting and linting
- unit tests
- data-contract tests
- security checks
- Docker build

### Training Workflow

- validate the dataset
- run feature generation
- train and evaluate the model
- log metrics to MLflow
- publish the model artifact
- require approval when production promotion is requested

### Deployment Workflow

- build and scan the image
- deploy to staging
- run health and prediction smoke tests
- deploy to production after gates pass
- keep the previous version ready for rollback

## 16. Implement Monitoring

Monitor four categories of signals.

### Data Quality

- missing-value rate
- invalid category rate
- duplicate rate
- schema changes
- late-arriving records
- feature range violations

### Data Drift

Compare production feature distributions with the training reference using Evidently or a similar tool. Track drift by feature, product category, region, and customer segment.

### Model Performance

When labels become available, calculate:

- forecast MAE, RMSE, and weighted MAPE
- recommendation precision, recall, NDCG, and conversion rate
- performance by category, region, and customer segment

### Service Health

Track:

- request count
- error rate
- p50 and p95 latency
- CPU and memory usage
- queue depth
- fallback rate
- model load failures

Send metrics to Prometheus and dashboards to Grafana. Store detailed drift and evaluation reports in object storage.

## 17. Handle Data Drift

Use a scheduled monitoring DAG to compare recent production data with the training reference.

### Mild Drift

If drift is below the action threshold:

1. Record the drift report.
2. Notify the data-science channel.
3. Continue serving the current model.
4. Increase monitoring frequency if drift is persistent.

### Significant Drift

If drift exceeds the agreed threshold:

1. Mark the affected features and segments.
2. Run data-quality and pipeline checks to rule out an upstream data error.
3. Freeze automatic promotion of new models until the issue is understood.
4. Trigger a retraining or re-experimentation workflow.
5. Compare the candidate model against the current production model.
6. Promote only if the candidate passes all quality and business gates.
7. Roll back or retain the current model if the candidate performs worse.

A drift alert should not automatically deploy a new model. Drift is a signal to investigate and evaluate.

## 18. Handle Additional Annotated Data

When new labelled or annotated data becomes available:

1. Upload it to versioned storage.
2. Validate it against the annotation schema.
3. Check label quality, duplicates, conflicts, and class balance.
4. Create a new DVC data version.
5. Add the new data to a training snapshot.
6. Trigger the training pipeline.
7. Compare the new candidate with the current production model.
8. Register and deploy only if the candidate passes the promotion gates.
9. Record the annotation version in MLflow and the model lineage metadata.

For recommendation data, use clicks, purchases, returns, and explicit feedback as labels where appropriate. For forecasting, append the newly observed demand period and recompute features without changing historical facts.

## 19. Add Tests Before Production

At minimum, create these test groups:

- schema and data-quality tests
- feature-generation tests
- time-split and leakage tests
- model-training tests
- prediction-shape and range tests
- API contract tests
- batch-inference tests
- monitoring threshold tests
- deployment smoke tests

Include a test that confirms the same feature code produces compatible training and inference columns.

## 20. Suggested End-to-End DAGs

Use separate DAGs with clear ownership:

### Daily Data DAG

```text
extract -> validate -> clean -> feature tables -> quality report
```

### Training DAG

```text
load version -> build features -> train -> evaluate -> register -> deploy staging
```

### Inference DAG

```text
load production model -> generate predictions -> write outputs -> publish metrics
```

### Monitoring DAG

```text
load reference and recent data -> quality checks -> drift report -> alert or retraining trigger
```

## 21. Recommended Local Build Order

Implement the system in this order so each stage remains testable:

1. Create the repository and environment.
2. Define schemas and sample data.
3. Build and test data cleaning.
4. Build the forecasting baseline.
5. Build the recommendation baseline.
6. Add feature generation.
7. Add time-based evaluation and business metrics.
8. Add MLflow experiment tracking.
9. Add DVC versioning.
10. Orchestrate ingestion and training with Airflow.
11. Build the FastAPI inference service.
12. Add batch inference.
13. Containerise the service.
14. Add CI/CD checks.
15. Add data, model, and service monitoring.
16. Test drift alerts and retraining with a deliberately shifted dataset.
17. Document rollback, ownership, and on-call actions.

## 22. Definition of Done

The solution is ready for demonstration when:

- Both models beat their defined baselines.
- Training can be reproduced from a Git commit and DVC data version.
- Every experiment and registered model has traceable metadata.
- Airflow can run ingestion, training, inference, and monitoring workflows.
- The API returns validated predictions and a model version.
- Batch predictions are stored and auditable.
- CI/CD runs tests before deployment.
- Drift and data-quality alerts are visible in a dashboard.
- Additional annotated data triggers a reproducible retraining workflow.
- A failed deployment can be rolled back to the previous model.
