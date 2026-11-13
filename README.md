# E-commerce forecasting and recommendations

An Olist MLOps project with Airflow execution, DVC file versioning, RustFS object
storage, MLflow experiment tracking, and gated model activation. Forecasting uses
next-day product/state demand; recommendation models use historical customer
snapshots. The managed workflow supports one shared local workspace on macOS/Linux.
It is not a distributed storage or high-availability deployment by itself.

## Responsibilities and layout

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

## Installation

Use Python 3.12, Git, Docker Engine and Docker Compose. The process lock uses
Unix `fcntl`; on Windows use a Linux environment such as WSL. Native library
requirements, including those needed by LightGBM, must also be available.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
# Exact installed Python environment used for release work:
python -m pip install -r requirements-runtime.lock.txt
python -m pip check
```

`requirements.txt` groups direct dependencies and retains supporting pins. It also
includes optional notebook tools that are not part of the installed release lock:

```bash
# Development / notebooks:
python -m pip install -r requirements.txt
```

`notebooks/requirements.txt` is the smaller notebook-only environment, not the full
workflow. Explore `notebooks/simple_baselines.ipynb` after preparing data/features.

Release preparation, training and activation validate exact installed versions
against `requirements-runtime.lock.txt`. Manifests also record Python/platform,
the installed distribution list and its digest, requirements hash, Git commit,
source-tree digest and optional `OLIST_IMAGE_DIGEST`. The lock reflects this tested
Python environment; validate native dependencies and regenerate/test a lock or
container image deliberately when moving to a different production platform.
Installing additional tools changes the environment digest: use the same release
environment for training and activation.

Commit code/configuration before starting a release workflow. Release mode is the
default; a dirty checkout is rejected. Development mode is explicitly marked and
its checkpoints cannot feed release training; its models cannot be activated.

## Storage

Start Docker, then initialize RustFS:

```bash
python scripts/storage/start_rustfs.py
```

`storage/compose.yaml` runs the official `rustfs/rustfs:1.0.0` image with persistent
data/log volumes. The setup creates bucket `olist-dvc` and configures DVC remote
`rustfs` at `s3://olist-dvc/cache`.

- Console: <http://127.0.0.1:19001>
- S3 endpoint: <http://127.0.0.1:19000>
- Private service credentials: `storage/.env`
- Private DVC credentials: `.dvc/config.local`

Both credential files are Git-ignored. Override local ports using
`RUSTFS_API_PORT` / `RUSTFS_CONSOLE_PORT` in `storage/.env`.

```bash
# Stop without deleting stored objects; do not add -v unless deletion is intended.
docker compose --env-file storage/.env -f storage/compose.yaml down
```

DVC uses `reflink,copy`: copy-on-write links where supported, physical copies
otherwise. There is no custom second full dataset under `data/snapshots/`.
DVC still needs local cache/workspace storage; reflink availability is filesystem
specific. Never replace these with ordinary writable hard links casually.
See [DVC cache optimization](https://doc.dvc.org/user-guide/data-management/large-dataset-optimization).

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

## Preparation and metadata publication

Configure Kaggle credentials for the worker with `~/.kaggle/kaggle.json` or
`KAGGLE_API_TOKEN`. Default paths, replay dates and model settings are in
`configs/settings.yaml`. Start a local development Airflow instance from the root:

```bash
export AIRFLOW_HOME="$PWD/airflow"
export AIRFLOW__CORE__DAGS_FOLDER="$PWD/pipelines"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
airflow standalone
```

Trigger `olist_data_preparation`. Its tasks are:

```text
begin_preparation → download → validate → publish_raw_version
  → clean → curate → forecasting_features → recommendation_features
  → publish_prepared_version → release_workspace
```

Processing tasks use temporary stage outputs. A task promotes its outputs only
after successful execution; a failure retains previous working outputs and cannot
publish a completed checkpoint. Preparation receipts enforce stage order. Raw
hashes are checked again before prepared publication to establish the parent link.
Upload must succeed before a new live pointer/catalog entry is published.

A version becomes consumable only after its metadata is committed through the
normal review/release process. Airflow does not commit or push Git automatically:

```bash
# After preparation succeeds, review the changes and record all checkpoint metadata.
git add snapshots/raw snapshots/prepared snapshots/catalog
git commit -m "Publish raw and prepared Olist dataset versions"
# Push the commit to your team's Git remote through the normal release process.
```

Read the prepared version ID from the `publish_prepared_version` task output or
its catalog filename. The preparation manifest records its producing run ID.

For fresh Kaggle files, use the download task's managed CLI with `--force`, or
refresh `data/raw/` only while no workflow owns the workspace, then run the whole
preparation DAG. Do not train from an incompletely prepared working directory.

## Training and model release

Trigger `olist_model_training` with the published prepared ID:

```bash
airflow dags trigger olist_model_training \
  --conf '{"prepared_version":"REPLACE_WITH_64_CHARACTER_PREPARED_VERSION"}'
```

The trigger form exposes row limits, estimator counts, seed, replay cutoffs,
candidate limits and Top K. Defaults come from YAML. Its tasks are:

```text
select_prepared_version
  → forecasting_baselines → recommendation_baselines
  → registered candidate jobs → compare_models
  → preserve_run_snapshot → backup_mlflow → release_workspace
```

Every job reads the selected working input version under the shared lease. The run
manifest resolves the paths through that version's catalog; no training input
files are copied. Each job checks the hashes of the resource it consumes.

Artifacts are isolated under `data/runs/<hash-of-run-id>/training/<task>/<model>/`.
Job indexes identify completed attempts; comparison uses their saved reports and
models. Run manifests and MLflow tags record raw/prepared IDs, code provenance,
environment and actual resolved parameters. A new training attempt requires a new
run ID. A preserved run cannot be modified in place.

Comparison selects on validation and uses test only as a release veto. It prepares
serving-ready bundles under `selection/models/` and reports under `selection/`;
it does not change `models/`. The DVC run checkpoint preserves models, all training
artifacts, manifests and decisions. The MLflow database is backed up separately.

After training succeeds, review and commit that run's uploaded metadata:

```bash
git add snapshots/runs/RUN_KEY
git commit -m "Publish evaluated Olist model run"
```

Then trigger the separate activation DAG:

```bash
airflow dags trigger olist_model_activation \
  --conf '{"training_run_id":"ORIGINAL_TRAINING_RUN_ID"}'
# Equivalent single operation:
python scripts/models/activate_models.py --pipeline-run-id ORIGINAL_TRAINING_RUN_ID
```

Activation requires committed run metadata, complete lineage, unchanged uploaded
artifact hashes, a clean matching source tree, and the same Python environment.
Development runs are rejected. An approved candidate replaces the active model;
a rejected candidate retains an existing model or installs the selected baseline
on the first deployment. Managed inference is locked during activation, and a
failed activation restores previous active model bytes.

Operational records in `data/activations/` describe the actual active selection
and file hash for each task, including retained older models. They are separate
from immutable training artifacts; back them up with operational state.

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

After activation, trigger `olist_batch_inference`, or specify inputs explicitly:

```bash
python scripts/inference/run_batch.py --task forecasting --as-of 2018-08-01 \
  --input-path data/features/forecasting/inference.parquet --models-dir models \
  --output-dir data/predictions
```

Recommendation inference takes `data/warehouse/olist.duckdb` as `--input-path`.
Predictions include selected model and training run IDs. There is no live API
serving or automatic monitoring/retraining deployment implemented here.

## Models and evaluation

`configs/production_models.json` registers candidate modules and maps their
parameters. Airflow creates one task per registered model; `run_job.py` calls its
shared `run_pipeline()` function. Add an implementation under
`src/models/<task>/`, its CLI entry point under `scripts/models/<task>/`, and a
registry entry. Reuse shared datasets/evaluation and `common.save_run()`.

Forecasting exposes `predict(feature_frame)` and predicts demand for the next day.
Features are past-only; labels crossing split boundaries are purged. Evaluation
uses full validation/test periods. `always_zero` is diagnostic and excluded from
baseline selection. Candidates must improve validation MAE without worse RMSE and
must produce varying predictions. Test can veto the selected winner; it does not
choose another candidate.

Recommendations expose `predict(snapshot, customer_id, k)`. Historical training,
validation, test and monitoring periods are separate; availability cutoffs exclude
future deliveries/history. Candidates must improve Recall@K without worse
Precision@K. Cold-start customers use popularity fallback. Defaults reserve August
2018 for inference/monitoring; they do not represent a live inventory system.

MLflow stores matching evaluation fingerprints and grouped run IDs, validation/test
metrics, parameters, artifacts and comparison decisions. View local experiments:

```bash
mlflow ui --backend-store-uri sqlite:///data/baselines/mlflow.db --port 5000
```

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
