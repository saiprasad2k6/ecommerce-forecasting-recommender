-- Connect DBeaver to data/warehouse/predictions.duckdb.
-- Disconnect that connection while Airflow/CLI loads additional batches.

-- Available batches. New outputs record actual generation time; legacy files
-- may use logical Airflow time or file modification time (see timestamp_source).
SELECT run_key, pipeline_run_id, as_of, run_timestamp, loaded_at,
       json_extract_string(manifest, '$.run_timestamp_source') AS timestamp_source
FROM prediction_runs
ORDER BY run_timestamp DESC;

-- Highest demand predictions from the latest batch.
SELECT f.product_id, f.category, f.customer_state, f.state_name, f.region,
       f.target_date, f.prediction, f.model_name
FROM forecasts f
WHERE f.run_key = (
    SELECT run_key FROM prediction_runs ORDER BY run_timestamp DESC, loaded_at DESC LIMIT 1
)
ORDER BY f.prediction DESC, f.product_id
LIMIT 100;

-- Predicted demand by state/category for one batch.
SELECT state_name, category, sum(predicted_demand) AS predicted_demand
FROM forecast_summary
WHERE run_key = 'REPLACE_WITH_RUN_KEY'
GROUP BY state_name, category
ORDER BY predicted_demand DESC;

-- Customers available in a batch.
SELECT customer_unique_id, k, model_name
FROM recommendation_predictions
WHERE run_key = 'REPLACE_WITH_RUN_KEY'
ORDER BY customer_unique_id
LIMIT 100;

-- Ranked, enriched customer recommendations.
SELECT customer_unique_id, customer_state, state_name, customer_city,
       rank, product_id, category, product_weight_g, model_name
FROM recommendations
WHERE run_key = 'REPLACE_WITH_RUN_KEY'
  AND customer_unique_id = 'REPLACE_WITH_CUSTOMER_ID'
ORDER BY rank;

-- Recommendation frequency, not predicted sales or model scores.
SELECT product_id, category, sum(recommendation_count) AS recommendations
FROM recommendation_summary
WHERE run_key = 'REPLACE_WITH_RUN_KEY'
GROUP BY product_id, category
ORDER BY recommendations DESC
LIMIT 20;
