# Model monitoring design and implementation procedure

This guide describes how to add monitoring to the Olist forecasting and recommendation project. It is a proposed implementation plan; the monitoring jobs and interfaces below are not yet implemented.

## 1. Keep training splits separate from monitoring data

The existing chronological training, validation and test datasets remain model-development artifacts:

| Dataset | Current purpose |
|---|---|
| Training | Fit model parameters using historical examples |
| Validation | Control early stopping, select the strongest baseline and assess the candidate |
| Test | Check the frozen candidate before publication |

Monitoring adds dated inference inputs, saved predictions and actual outcomes. It does not require another permanent split of every training dataset.

```text
Data preparation → Features → Training and baseline jobs → Comparison → Active model
                                                                          ↓
Dated inputs → Batch inference → Saved predictions → Outcomes → Monitoring
```

The forecasting pipeline currently reads split Parquet files. The recommender builds its training and evaluation snapshots directly from the warehouse. Preserve this distinction when adding monitoring.

Batch inference is implemented in `pipelines/batch_inference.py`. `pipelines/monitoring_pipeline.py` is still an empty placeholder. Shared replay and proposed monitoring defaults live in `configs/settings.yaml`.

## 2. Reserve a historical replay period

Olist is a fixed historical dataset. Simulate production by releasing a model at a historical date and advancing through later dates without retraining it each day.

Use a separate replay configuration and output directory. One possible timeline is:

| Purpose | Proposed period |
|---|---|
| Training examples | Before June 1, 2018 |
| Validation and early stopping | June 2018 |
| Test and release decision | July 2018 |
| Monitoring simulation | August 2018, after the release decision |

These replay boundaries are now the configured defaults. Rebuild the feature files and retrain before using them. Forecasting development splits use next-day target dates; inference.parquet holds August feature dates without requiring future labels.

For the proposed recommendation replay, use June 1 as the validation cutoff, July 1 as the test cutoff and August 1 as the test-period end. Keep the baseline and candidate evaluation settings identical.

Forecasting boundaries must respect the target date. A July 31 feature row predicts August 1 demand, so its outcome belongs to August. Exclude rows whose next-day labels cross the relevant training or evaluation boundary.

### Replay procedure

1. Build training and evaluation data using the replay cutoffs and information available at those cutoffs.
2. Run the separate baseline and candidate jobs with a shared `pipeline_run_id`.
3. Run comparison and identify the selected model or baseline fallback.
4. Freeze that exact model artifact and its reference data.
5. Begin simulated inference after release. For example, an August 1 inference run uses observations through August 1 to predict August 2 demand.
6. Advance `as_of` by one day for each subsequent inference run.
7. Score predictions only after their outcomes are sufficiently complete.

Use the simulated `as_of` date consistently for history, product availability, customer selection, feature calculation and outcome eligibility. Do not use the computer's current date as the historical cutoff.

## 3. Separate inference features from labels

The forecasting feature builder uses `LEAD(demand, 1)` for development labels. It retains valid feature rows without future labels for inference.parquet, and drops the target column from that file.

When extending feature preparation, ensure that:

- Lag and rolling features are calculated using observations available through `as_of`.
- Training preparation attaches next-day labels and excludes missing labels.
- Inference preparation keeps valid feature rows even when future labels are unavailable.
- Product/state series are created from the population known by the simulated cutoff, rather than from future product/state combinations.
- At least 28 prior days are available for the existing lag and rolling features.

Reuse the existing formulas and model encoders. Avoid maintaining independent versions of the same feature calculation for training and inference.

For recommendations, reuse `RecommendationSnapshot(history, cutoff)`. Select customers using information available at inference time. Do not select the inference cohort from future purchase labels.

## 4. Save predictions at inference time

Write dated Parquet partitions:

```text
data/predictions/
    forecasting/as_of=2018-08-01/predictions.parquet
    recommendations/as_of=2018-08-01/predictions.parquet
```

Suggested forecasting fields:

```text
as_of
target_date
product_id
customer_state
prediction
active_model_run_id
pipeline_run_id
```

Suggested recommendation fields:

```text
as_of
customer_unique_id
recommended_products  # Ordered list; preserve recommendation rank
k
horizon_days
active_model_run_id
pipeline_run_id
```

Also retain the input features used for inference, or an immutable snapshot reference from which they can be recovered. Prediction outputs alone are insufficient for input-drift analysis.

`pipeline_run_id` identifies the inference execution. `active_model_run_id` identifies the exact model that generated the predictions. Neither should be inferred later from whichever active `.pkl` file is currently present.

Make writes idempotent: a retry must not duplicate predictions. Use keys containing the entity, inference date, forecast target or recommendation horizon, and model version. Preserve predictions from different model versions instead of overwriting their history.

## 5. Version the active model and its reference

When comparison publishes a model, save metadata alongside the active artifact containing:

- The selected artifact's MLflow run ID.
- The originating training `pipeline_run_id`.
- The model's effective release date.
- The reference dataset version.
- The feature schema or feature-code version.

If comparison publishes a baseline fallback, record the baseline's run ID. If it retains an existing model, retain that model's identity and reference as well.

Store a reference snapshot for each active model:

```text
data/reference/<active_model_run_id>/
    features.parquet
    metrics.json
    metadata.json
```

Use a representative sample from release evaluation. Do not use the forecasting training sample without accounting for its deliberate positive/zero-demand sampling.

Reference metrics must use the same population, outcome eligibility rules and horizon as monitoring. For example, calculate a seven-day recommendation reference score if monitoring uses seven-day purchase outcomes; the current monthly evaluation score is not directly interchangeable with it.

Freeze the reference for that model version. Do not silently replace it every day, which could hide gradual changes.

## 6. Run three types of monitoring

| Area | Initial checks |
|---|---|
| Data quality | Missing required features, duplicate keys, stale inputs, incomplete date partitions |
| Input drift | Changes in demand and rolling features, state mix, customer frequency and unseen-product rate |
| Performance | Forecasting MAE/RMSE and recommendation Precision@K/Recall@K |

Start with a rolling seven-day input window and expand it when volumes are too small. Account for normal weekday and seasonal patterns when comparing windows.

Evidently can compare current features with the frozen reference and produce a drift report. Input drift is a signal to investigate; it does not by itself prove that prediction quality deteriorated.

### Forecasting outcomes

1. Load saved predictions for target dates whose source data is complete.
2. Join actual demand on `target_date`, `product_id` and `customer_state`.
3. Calculate MAE and RMSE with scikit-learn.
4. Report how many predictions have mature, matched outcomes.

Do not treat an incomplete or missing ingestion partition as zero demand. In the historical replay, reveal actual demand only when it would have been available.

### Recommendation outcomes

1. Choose a fixed outcome horizon, such as seven days after recommendation.
2. Keep the original ranked recommendation lists.
3. Wait for the outcome window and required delivery information to become available.
4. Join eligible purchases to the saved recommendations.
5. Calculate Precision@K and Recall@K using the existing ranking-metric definitions.

Document which customers enter each performance average. The existing offline evaluator averages over purchasing customers; an inference cohort can also contain non-purchasers. Keep reference and current evaluation cohorts consistent, and report cohort counts and outcome coverage. A horizon that has not matured must not be scored as having no purchases.

Calculate performance from saved predictions. Rerunning the latest active model would evaluate different predictions if the model changed after the original inference date.

## 7. Add separate inference and monitoring DAGs

Suggested implementation structure:

```text
src/inference/batch.py
scripts/inference/run_batch.py

src/monitoring/monitor.py
scripts/monitoring/run_monitoring.py

pipelines/batch_inference.py
pipelines/monitoring_pipeline.py
```

Keep reusable logic in `src/` and thin CLI wrappers in `scripts/`, following the existing project structure. Use the current model `predict()` methods, feature preparation, ranking metrics and standard libraries rather than duplicating model logic.

### Batch-inference DAG

```text
Check source completeness
→ Build dated inference inputs
→ Load active models and metadata
→ Save predictions and model identities
```

### Monitoring DAG

```text
Load recent inference inputs and saved predictions
→ Check data quality
→ Compare current inputs with the model's reference
→ Attach mature actual outcomes
→ Calculate performance metrics
→ Log reports and alert status
```

Initially trigger these DAGs manually with a historical `as_of` parameter. Once replay is verified, a real daily schedule should derive data cutoffs from the intended data interval and completeness rules.

The two jobs can run at different times: input quality and drift checks can run before outcomes are available, while performance checks wait for mature labels. Monitoring should not rebuild training splits or retrain models by default.

## 8. Track monitoring in MLflow

Use separate experiments:

```text
ecommerce-forecasting-monitoring
ecommerce-recommendation-monitoring
```

Record these tags or parameters for each monitoring run:

```text
pipeline_run_id
active_model_run_id
window_start
window_end
reference_version
outcome_horizon
```

Log MAE/RMSE or Precision@K/Recall@K, evaluated row/customer counts, outcome coverage, the drift report and alert status. Reuse the existing tracking database. This provides a monitoring timeline linked to the model that actually served predictions.

Partition monitoring outputs by model version if a window contains predictions from multiple models. Do not aggregate their performance as if a single model produced all predictions.

## 9. Define alerts before automating retraining

The ambiguous `monitoring.retraining_threshold` setting has been removed. Automatic retraining is disabled. Calibrate named metric thresholds, directions and persistence periods before enabling alerts or automatic retraining.

Start with:

- Data-quality alerts for missing or incomplete inputs.
- Drift alerts for changes in selected feature distributions.
- Performance alerts for persistent degradation against a comparable reference.

For example, a rule might flag MAE more than 20% above its reference for three consecutive eligible windows. This is an illustrative starting rule, not a universal threshold. Define a separate absolute rule when the reference metric is zero, and require enough mature outcomes before assessing performance.

Use replay results to calibrate thresholds. Treat overlapping rolling windows as related observations rather than independent evidence.

Investigate an alert before triggering training. A data-pipeline failure requires a data fix, not necessarily a new model. Retraining should continue to use the existing baseline, candidate and comparison workflow before any replacement is published.

## 10. Implement and validate in stages

1. **Historical replay:** configure cutoffs and verify that no future inputs or outcomes enter an earlier stage.
2. **Active-model metadata:** record the selected model identity and preserve it when publication retains an existing model.
3. **Saved inference:** write dated features and predictions with retry-safe keys.
4. **Performance monitoring:** join mature outcomes and calculate the existing basic metrics.
5. **Reference and drift:** freeze representative references and add Evidently reports.
6. **Orchestration and alerts:** fill the DAG placeholders, log monitoring runs and calibrate alert rules.

Validate with temporary sample data before running against the project warehouse. Include checks for:

- Predictions and features remaining unchanged when later transactions are added.
- Correct next-day target boundaries and recommendation delivery cutoffs.
- Missing source partitions remaining distinct from genuine zero demand.
- Repeated inference runs not duplicating stored predictions.
- Missing or immature outcomes being deferred instead of scored as zero.
- Monitoring remaining linked to the original serving model after a model replacement.
- Small hand-calculated examples matching the reported performance metrics.

## References

- [Evidently data-drift documentation](https://docs-old.evidentlyai.com/presets/data-drift): reference/current dataset comparison and drift reports. Use the API matching the project's installed Evidently version when implementing.
- [MLflow Tracking APIs](https://mlflow.org/docs/latest/ml/tracking/tracking-api/): logging monitoring metrics, tags and artifacts.
