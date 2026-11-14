# E-commerce forecasting and recommendations

This project uses the Brazilian Olist e-commerce dataset to answer two questions:
how much demand should we expect tomorrow for each product and customer state,
and which products should we recommend to a customer based on their history?

It covers the workflow from downloading raw data to generating batch predictions.
Airflow runs preparation, training, activation and inference. LightGBM candidates
are evaluated against simple baselines before they can become active models.
MLflow records experiments; DVC and RustFS preserve datasets and model artifacts;
Git records code, configuration and version metadata so a run can be traced and
restored.

The default dates replay historical data: validation starts in June 2018, testing
in July, and inference/monitoring inputs in August. This is a local MLOps workflow
for one shared workspace on macOS/Linux. Live API serving, monitoring jobs and
automatic retraining are not implemented.

## Environment setup

### 1. Install prerequisites and Python dependencies

Use Python 3.12, Git, Docker Engine with Docker Compose, and any native libraries
required by LightGBM. On Windows, use a Linux environment such as WSL: workspace
locking depends on Unix `fcntl`.

Run all commands below from the project root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-runtime.lock.txt
python -m pip check
```

The runtime lock reproduces the Python environment used for release work.
For notebook development, `requirements.txt` also includes optional notebook tools:

```bash
python -m pip install -r requirements.txt
```

Use the runtime lock for release runs. Activate `.venv` in each new terminal.

### 2. Configure Kaggle access

Provide Kaggle credentials using `~/.kaggle/kaggle.json` or the
`KAGGLE_API_TOKEN` environment variable. They must be available to the Airflow
worker that downloads `olistbr/brazilian-ecommerce`.

Review `configs/settings.yaml` for dataset paths, replay dates, training limits,
estimator counts and recommendation settings. Candidate models are registered in
`configs/production_models.json`.

### 3. Configure DVC and start RustFS

The checkout normally already contains `.dvc/`. Only if DVC has not been
initialized, run:

```bash
dvc init
dvc remote add -d rustfs s3://olist-dvc/cache
dvc remote modify rustfs region us-east-1
dvc config cache.type reflink,copy
dvc config core.analytics false
```

Ensure `.dvc/.gitignore` retains its generated entries and also includes:

```gitignore
/workspace.json
/workspace.lock
/snapshot.lock
```

Start Docker Desktop or Docker Engine, then run:

```bash
python scripts/storage/start_rustfs.py
dvc version
dvc remote list
docker compose --env-file storage/.env -f storage/compose.yaml ps
```

The setup script starts `rustfs/rustfs:1.0.0`, creates bucket `olist-dvc`, and
configures the `rustfs` DVC remote at `s3://olist-dvc/cache`. It generates or reuses
private credentials in `storage/.env` and writes private DVC settings to
`.dvc/config.local`. Both files are Git-ignored. The `permissions` container
exiting with code 0 is expected; it sets volume ownership before RustFS starts.

| Service | Address |
|---|---|
| RustFS console | <http://127.0.0.1:19001> |
| RustFS S3 endpoint | <http://127.0.0.1:19000> |

Use the credentials in `storage/.env` to access the console. To change ports, set
`RUSTFS_API_PORT` / `RUSTFS_CONSOLE_PORT` in that file and rerun the setup script.
Commit shared DVC configuration changes through the normal review process; keep
credential files private.

## Run the project in order

Follow this sequence for a release that can be activated. Release preparation and
training require a clean Git checkout, so review and commit your intended code,
configuration and README changes before starting. Wait for each DAG to succeed
before moving to the next step.

For experiments with uncommitted source, preparation and training accept
`--conf '{"release":false}'` (include `prepared_version` for training). Development
training still needs committed input metadata, and development runs cannot be
activated. Use the release path below when you need batch predictions.

### 1. Start Airflow — terminal 1

```bash
source .venv/bin/activate
export AIRFLOW_HOME="$PWD/airflow"
export AIRFLOW__CORE__DAGS_FOLDER="$PWD/pipelines"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
airflow standalone
```

Keep this terminal running. Open the Airflow address printed at startup and sign
in using the generated credentials. This starts the local development Airflow
services and loads the DAGs from `pipelines/`.

### 2. Prepare a command terminal — terminal 2

```bash
source .venv/bin/activate
export AIRFLOW_HOME="$PWD/airflow"
export AIRFLOW__CORE__DAGS_FOLDER="$PWD/pipelines"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
airflow dags list
```

Wait for the project DAGs to appear. If a DAG is paused, unpause it in the UI or
with the relevant `airflow dags unpause` command below.

### 3. Prepare the data

```bash
airflow dags unpause olist_data_preparation
airflow dags trigger olist_data_preparation
```

Follow the run in the Airflow UI. It downloads or reuses the raw CSVs, validates
and cleans them, builds the warehouse and features, and uploads raw and prepared
versions to RustFS through DVC.

After the DAG succeeds, find the 64-character prepared version ID in the
`publish_prepared_version` task log or the corresponding filename under
`snapshots/catalog/prepared/`. Select the version produced by this run.

### 4. Publish the dataset metadata to Git

Review the generated changes, then commit the raw/prepared pointers and catalogs
together:

```bash
git diff -- snapshots/raw snapshots/prepared snapshots/catalog
git status --short
git add snapshots/raw snapshots/prepared snapshots/catalog
git commit -m "Publish raw and prepared Olist dataset versions"
```

Airflow uploads file contents but does not commit or push Git metadata. Publish
the commit to your team's remote through the normal release process. Training
requires the selected version's metadata to be committed in Git HEAD.

### 5. Train and evaluate models

Replace the placeholder with the prepared version from step 3:

```bash
airflow dags unpause olist_model_training
airflow dags trigger olist_model_training \
  --conf '{"prepared_version":"REPLACE_WITH_64_CHARACTER_PREPARED_VERSION"}'
```

Wait for success. Record the original Airflow training run ID shown in the UI;
activation uses this exact ID. The workflow evaluates baselines, trains registered
candidates, compares them, uploads the run artifacts and backs up MLflow.
Training does not replace the active models.

The trigger form also exposes row limits, estimator counts, seed, date cutoffs,
candidate limits and Top K. Defaults come from `configs/settings.yaml`; split
parameters must match the selected prepared version.

### 6. Inspect the results in MLflow — terminal 3

```bash
source .venv/bin/activate
mlflow ui --backend-store-uri sqlite:///data/baselines/mlflow.db --port 5000
```

Open <http://localhost:5000>. Inspect parameters, validation/test metrics,
artifacts and comparison decisions in `ecommerce-demand-forecasting` and
`ecommerce-recommendations`. Keep this terminal running while using the UI;
press `Ctrl+C` to stop it. You can also start MLflow while training is running.

### 7. Publish the model-run metadata

Return to terminal 2. Find the matching run manifest under
`snapshots/runs/<run-key>/manifest.json`: the run key is a hash of the training
run ID, not the original Airflow ID. Review that run's metadata, then replace
`RUN_KEY` below:

```bash
git status --short
git add snapshots/runs/RUN_KEY
git commit -m "Publish evaluated Olist model run"
```

Publish the commit through the normal release process before activation.

### 8. Activate the evaluated model selection

Replace the placeholder with the original training run ID from step 5:

```bash
airflow dags unpause olist_model_activation
airflow dags trigger olist_model_activation \
  --conf '{"training_run_id":"ORIGINAL_TRAINING_RUN_ID"}'
```

Wait for success. Activation checks committed metadata, lineage, artifact hashes,
source and environment before updating `models/`. A candidate that passes the
gates replaces the active model. A rejected candidate retains an existing model,
or installs the selected baseline on the first deployment.

The equivalent single operation, when Airflow orchestration is unnecessary, is:

```bash
python scripts/models/activate_models.py --pipeline-run-id ORIGINAL_TRAINING_RUN_ID
```

### 9. Generate batch predictions

```bash
airflow dags unpause olist_batch_inference
airflow dags trigger olist_batch_inference
```

This runs both forecasting and recommendation inference with active models.
Defaults use `as_of=2018-08-01`, forecasting inputs from
`data/features/forecasting/inference.parquet`, and recommendation history from
`data/warehouse/olist.duckdb`. Override dates and paths in the trigger form for
another batch.

Outputs are written under:

```text
data/predictions/<task>/as_of=<date>/model_run_id=<id>/predictions.parquet
```

Rows include the selected model and training run IDs. Forecasting uses features
on `as_of` to predict the next day's demand. Recommendations use history before
`as_of`.

To run only forecasting directly:

```bash
python scripts/inference/run_batch.py --task forecasting --as-of 2018-08-01 \
  --input-path data/features/forecasting/inference.parquet --models-dir models \
  --output-dir data/predictions
```

For recommendation inference, use `--task recommendation` and
`--input-path data/warehouse/olist.duckdb`.

### 10. Stop local services when finished

Stop Airflow and MLflow with `Ctrl+C` in their terminals. Stop RustFS without
deleting its stored objects:

```bash
docker compose --env-file storage/.env -f storage/compose.yaml down
```

Do not add `-v` unless you intend to delete the storage volumes. On subsequent
starts, reactivate `.venv`, rerun `python scripts/storage/start_rustfs.py`, and
start Airflow and MLflow as above. Existing versions can be restored with
`dvc pull` after checking out their Git metadata and configuring storage.

## What each stage does

### Data preparation: `olist_data_preparation`

| Stage | Purpose and output |
|---|---|
| `begin_preparation` | Reserves the shared workspace and records preparation parameters. |
| `download_olist_data` | Downloads Olist CSVs into `data/raw/`; complete local downloads are normally reused. |
| `validate_olist_data` | Checks raw files against `configs/schema.yaml` and writes `data/validation/raw.json`. |
| `publish_raw_version` | Uploads raw files and validation results through DVC and records their content-based version. |
| `clean_olist_data` | Produces cleaned working tables under `data/processed/`. |
| `prepare_curated_data` | Builds curated tables in `data/curated/` and the DuckDB warehouse at `data/warehouse/olist.duckdb`. |
| `build_forecasting_features` | Builds temporal forecasting features and split metadata under `data/features/forecasting/`. |
| `build_recommendation_features` | Writes an inspectable customer recommendation example under `data/features/recommendations/`; training examples are built from the warehouse. |
| `publish_prepared_version` | Verifies preparation receipts and raw hashes, then uploads forecasting inputs and the warehouse with links to the raw version. |
| `release_workspace` | Releases workflow ownership after success or failure. |

Stages promote temporary outputs only after success. Failed processing retains
previous working outputs; failed uploads cannot publish a completed checkpoint.
For fresh Kaggle files, use the managed download CLI with `--force`, or refresh
`data/raw/` only when no workflow owns the workspace, then rerun preparation.

### Training: `olist_model_training`

| Stage | Purpose and output |
|---|---|
| `select_prepared_version` | Reserves the workspace and verifies the explicitly selected committed prepared version, parent raw version, input hashes and split parameters. |
| `evaluate_forecasting_baselines` | Evaluates last-day, weekly, moving-average and seasonal rules; `always_zero` is diagnostic only. |
| `evaluate_recommendation_baselines` | Evaluates global popularity, category popularity and purchase-history rules. |
| `train_forecasting_lightgbm` | Trains the registered LightGBM demand candidate and evaluates validation/test periods. |
| `train_recommendation_lightgbm_ranker` | Trains the registered LightGBM ranking candidate on historical customer examples. |
| `compare_models` | Selects candidates on validation, applies the test veto, and creates serving bundles and decisions under the run's `selection/` directory. |
| `preserve_run_snapshot` | Uploads the immutable run artifacts through DVC and writes the run pointer and manifest. |
| `backup_mlflow` | Backs up the tracking SQLite database to `backups/mlflow/`. |
| `release_workspace` | Releases workflow ownership after success or failure. |

Candidate task names come from `configs/production_models.json`. Both preparation
and training also have a `workflow_failed` watcher so cleanup cannot hide an
upstream failure.

Training artifacts live under
`data/runs/<hash-of-run-id>/training/<task>/<model>/`. Job indexes identify
completed attempts; comparison uses those saved reports and models. Every new
training attempt needs a new run ID. Preserved runs cannot be modified in place.

### Activation and inference

`olist_model_activation` runs `activate_models`: it verifies the preserved run
and installs its selection in `models/`. Managed inference is locked during the
update; a failed activation restores the previous active model bytes. Operational
records under `data/activations/` identify the actual active model, including
older models retained after a candidate is rejected.

`olist_batch_inference` runs `predict_forecasting` followed by
`predict_recommendation`. These tasks load approved active bundles and write
Parquet predictions; they do not train or select models. Restored bundles can
serve predictions without the historical MLflow database.

### Evaluation rules

Forecasting predicts next-day demand using past-only features. Labels crossing
split boundaries are purged, and evaluation covers the full validation/test
periods. A candidate must improve validation MAE without worse RMSE and produce
varying predictions. `always_zero` cannot win baseline selection.

Recommendation models use separate historical training, validation, test and
monitoring periods. Availability cutoffs exclude future deliveries and customer
history. A candidate must improve Recall@K without worse Precision@K; cold-start
customers use popularity fallback. For both tasks, test can veto the selected
winner but does not choose another candidate.

## Repository and operations reference

The sections below describe file ownership, version identities, recovery and
advanced CLI usage after the main workflow is familiar.

## Repository layout

| Location/tool | Responsibility |
|---|---|
| `src/data_processing/` | Downloading, validation, cleaning and curation implementations |
| `src/features/`, `src/models/`, `src/inference/` | Reusable feature, model, evaluation and prediction logic |
| `scripts/` | CLI entry points for individual operations |
| `pipelines/` | Airflow task dependencies, retries and trigger parameters |
| `src/versioning/` | DVC publication, lineage, environment checks and workspace ownership |
| `data/` | Working datasets, run artifacts and operational activation records |
| `snapshots/` | Small manifests and DVC pointers; commit these to Git |
| `models/` | Currently active model bundles, outside Git |
| `storage/` | RustFS Compose configuration and ignored credentials |
| `tests/` | Data/model tests, failure cases, DAG contracts and DVC recovery tests |
| MLflow | Parameters, metrics, models, comparison decisions and dataset-version tags |
| DVC + RustFS | Content-addressed preservation and restoration of actual file bytes |
| Git | Code, configuration, runtime lock and published version metadata |

Airflow is the only workflow executor. There is no executable `dvc.yaml` pipeline
or separate DVC training runner. DVC is called at publication checkpoints.


## Versions and lineage

There are three meaningful checkpoints, not a version for every task:

```text
Raw version A
  └─ preparation run + code + parameters
       └─ Prepared version B
            └─ training run + code + parameters + MLflow runs
                 └─ Model version C
```

| Checkpoint | Actual local files | Git metadata |
|---|---|---|
| Raw | `data/raw/`, `data/validation/raw.json` | `snapshots/raw/*.dvc`, `snapshots/catalog/raw/<id>.json` |
| Prepared | `data/features/forecasting/`, `data/warehouse/olist.duckdb` | `snapshots/prepared/*.dvc`, `snapshots/catalog/prepared/<id>.json` |
| Model/run | `data/runs/<run-key>/` | `snapshots/runs/<run-key>/artifacts.dvc`, `manifest.json` |

CSV originals remain in `data/raw/`. Cleaned and curated working outputs live in
`data/processed/` and `data/curated/`; they are rebuildable intermediates, not
additional release checkpoints. Recommendation training constructs its historical
examples from the versioned warehouse. The single-customer output under
`data/features/recommendations/` is an inspectable example, not the ranker's
training table.

Version IDs are content-based. A prepared identity also includes its parent raw
version, preparation parameters and code commit. Raw manifests record the Kaggle
dataset identifier and observation time; observation time is not a claim that a
fresh download occurred (complete local downloads are normally reused).

The live `.dvc` pointers reference the normal working paths. Historical pointer
revisions are preserved in Git; catalog manifests retain the parent links and
exact DVC/file hashes. Always commit the raw/prepared pointers and catalog files
together. DVC does not infer Airflow lineage automatically.

Training requires an explicit 64-character `prepared_version`. It verifies:

- The prepared and parent raw manifests are committed in Git HEAD.
- The live prepared pointers match that selected version and are committed.
- Working input hashes and split dates match that version and requested parameters.
- Release inputs came from release preparation.

It never selects the newest folder. If another version is currently materialized,
use a checkout containing the desired release metadata and `dvc pull` first.
Multiple model runs can use the same prepared version without making another
physical dataset snapshot.


## Workspace isolation, retries and crash recovery

Preparation and training reserve `.dvc/workspace.json` for the entire workflow.
A process-level `.dvc/workspace.lock` serializes individual operations. This
prevents separate DAGs from interleaving writes and reads in the shared folders.
Managed preparation CLIs, model jobs, comparison, activation and batch inference
participate in the guard. Low-level Python functions are reusable primitives, not
a security boundary: do not mutate their files outside the managed workflow.

A conflicting workflow fails/retries rather than silently stealing ownership.
Both DAGs release ownership after success or failure. A separate failure watcher
ensures successful cleanup cannot make a failed workflow appear successful.

A killed worker/scheduler can leave a lease. It never expires automatically.
First stop the owning workflow and ensure its workers have exited. Inspect the
owner, then release only that exact workflow:

```bash
cat .dvc/workspace.json
python scripts/versioning/snapshot.py release-workspace \
  --kind prepare --pipeline-run-id ORIGINAL_PREPARATION_RUN_ID
# For training, use --kind train and the original training run ID.
```

Restart the workflow from its beginning after a lease has been released; do not
retry a model task without reselecting/verifying its version. A different owner
cannot release another workflow's lease.

This is a single-host/shared-filesystem contract, not a distributed lock service.
For concurrent production workers, use isolated workspaces materialized from
explicit DVC versions and a scheduler/service coordinating publication. Do not
mount unrelated local directories and assume this lease coordinates them.


## Individual CLI operations

The scripts expose the same operations as Airflow. A controlled preparation run
begins with `begin-preparation`. Each preparation command must run through
`execute-preparation --stage ... -- COMMAND`, in DAG order:

```bash
export OLIST_PARAMETERS='{"validation_start":"2018-06-01","test_start":"2018-07-01","monitoring_start":"2018-08-01","monitoring_end":"2018-09-01"}'
python scripts/versioning/snapshot.py begin-preparation --pipeline-run-id prep-001
python scripts/versioning/snapshot.py execute-preparation --pipeline-run-id prep-001 \
  --stage download_olist_data -- python scripts/data_processing/download_olist_data.py
python scripts/versioning/snapshot.py execute-preparation --pipeline-run-id prep-001 \
  --stage validate_olist_data -- python scripts/data_processing/validate_olist_data.py
python scripts/versioning/snapshot.py raw --pipeline-run-id prep-001
# Execute clean_olist_data, prepare_curated_data, build_forecasting_features and
# build_recommendation_features through the same managed wrapper; supply the
# feature/recommendation arguments matching OLIST_PARAMETERS.
python scripts/versioning/snapshot.py prepared --pipeline-run-id prep-001
python scripts/versioning/snapshot.py release-workspace --kind prepare --pipeline-run-id prep-001
```

After committing metadata, a manifest-backed model job is:

```bash
export OLIST_PARAMETERS='{"forecast_estimators":50,"ranking_estimators":50}'
python scripts/versioning/snapshot.py begin-training --pipeline-run-id train-001 \
  --prepared-version PREPARED_VERSION
python scripts/models/run_job.py --pipeline-run-id train-001 --task forecasting --model baselines
# Run recommendation baselines and every configured candidate, then:
python scripts/models/comparison/compare_models.py --pipeline-run-id train-001
python scripts/versioning/snapshot.py artifacts --pipeline-run-id train-001
python scripts/versioning/snapshot.py backup-mlflow
python scripts/versioning/snapshot.py release-workspace --kind train --pipeline-run-id train-001
```

Always release ownership in your shell/automation's failure cleanup too. Prefer
Airflow for the full workflow. `--development` is supported on begin operations;
Airflow exposes the equivalent `release=false`. Development still requires
committed input metadata but permits uncommitted source and cannot activate models.


## Recovery and lineage audit

Use a separate checkout for historical recovery rather than changing the source
of an active worker. Check out the Git release containing the source, raw/prepared
pointers, run pointer and manifests. Configure private storage credentials there,
install its runtime lock, then:

```bash
dvc pull
python scripts/versioning/snapshot.py lineage --pipeline-run-id ORIGINAL_TRAINING_RUN_ID
python scripts/models/activate_models.py --pipeline-run-id ORIGINAL_TRAINING_RUN_ID
```

`lineage` resolves model → prepared → raw and verifies committed metadata. It does
not download files or assert that a remote backup is healthy. `dvc pull` verifies
that the referenced data can actually be materialized. Activation additionally
checks the full local artifact hashes. Restoring models and running inference do
not require the old MLflow database.

## Adding a candidate model

`configs/production_models.json` registers candidate modules and maps their
parameters. Add an implementation under `src/models/<task>/`, a CLI entry point
under `scripts/models/<task>/`, and a registry entry. Airflow creates a task for
each registered model; `run_job.py` calls its shared `run_pipeline()` function.
Reuse shared datasets, evaluation and `common.save_run()`.

Forecasting models expose `predict(feature_frame)`; recommendation models expose
`predict(snapshot, customer_id, k)`.

## Storage durability, backups and retention

The local RustFS container and SQLite databases are development infrastructure.
For production, deploy durable object storage independently of workers, use TLS
and scoped credentials from your secret manager, and back up objects off-machine.
Configure the DVC endpoint/credentials in private local configuration, not source
code. A production tracking service also needs its own database/artifact backups.
No remote server, TLS certificate or off-machine backup destination is provisioned
by this repository.

`backup-mlflow` uses SQLite's backup API to include committed WAL contents. It
writes timestamped backups to `backups/mlflow/`, outside DVC/Git. Copy those backups,
MLflow artifacts and activation records to the configured backup destination and
test restoration. For a remote SQL server use its native backup process; the
SQLite backup command deliberately rejects other tracking URIs.

Keep Git release tags/commits and their DVC objects for the required retention
period. Do not run remote garbage collection against only today's workspace.
Catalog JSON alone is not a DVC garbage-collection root: retain the Git commits
containing each released live `.dvc` pointer. Garbage collection is an operator
operation, never an Airflow task; account for all retained commits, tags, branches
and repositories sharing the remote before deleting anything. See
[DVC GC](https://doc.dvc.org/command-reference/gc) and
[DVC remote storage](https://doc.dvc.org/user-guide/data-management/remote-storage).


## Verification and cleanup

```bash
python -m pip check
python -m unittest discover -s tests
```

Tests cover temporal evaluation, model gates, real DAG parsing and quoting,
workspace conflicts, failed stage promotion, failed uploads, uncommitted/dirty
release rejection, complete lineage, an empty-cache DVC restore, and inference
from restored bundles without MLflow history. By default DVC tests use temporary
local remotes. `OLIST_TEST_RUSTFS=1` uses the configured RustFS server and uploads
small fixtures under `olist-dvc/verification/`; retain or clean these test objects
according to your test-storage policy.

Run `scripts/maintenance/cleanup_data.py --help` or
`scripts/maintenance/reset_repository.py --help` before local cleanup. Stop all
workers and services using the workspace first. Removing local data does not
remove remote versions, but historical restoration still requires retained Git
metadata and DVC objects. Do not delete metadata to free dataset disk space.
