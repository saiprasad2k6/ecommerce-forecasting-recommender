"""Load validated prediction batches into a standalone DuckDB database for SQL analysis."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd

from src.common.config import SETTINGS
from src.versioning.snapshots import ROOT, identifier
from src.versioning.workspace import locked

PREDICTIONS_DATABASE = ROOT / SETTINGS["inference"].get("predictions_database", "data/warehouse/predictions.duckdb")
REFERENCE_FILES = ("products.csv", "categories.csv", "customers.csv", "orders.csv")
PROVENANCE = {"as_of", "model_name", "model_role", "model_run_id",
              "training_pipeline_run_id", "pipeline_run_id"}
REQUIRED = {
    "forecasting": PROVENANCE | {"purchase_date", "product_id", "customer_state", "target_date", "prediction"},
    "recommendation": PROVENANCE | {"customer_unique_id", "recommended_products", "k"},
}
# Brazilian federative units; forecast state means customer location, not seller location.
STATES = [
    ("AC", "Acre", "North"), ("AL", "Alagoas", "Northeast"),
    ("AP", "Amapá", "North"), ("AM", "Amazonas", "North"),
    ("BA", "Bahia", "Northeast"), ("CE", "Ceará", "Northeast"),
    ("DF", "Distrito Federal", "Central-West"), ("ES", "Espírito Santo", "Southeast"),
    ("GO", "Goiás", "Central-West"), ("MA", "Maranhão", "Northeast"),
    ("MT", "Mato Grosso", "Central-West"), ("MS", "Mato Grosso do Sul", "Central-West"),
    ("MG", "Minas Gerais", "Southeast"), ("PA", "Pará", "North"),
    ("PB", "Paraíba", "Northeast"), ("PR", "Paraná", "South"),
    ("PE", "Pernambuco", "Northeast"), ("PI", "Piauí", "Northeast"),
    ("RJ", "Rio de Janeiro", "Southeast"), ("RN", "Rio Grande do Norte", "Northeast"),
    ("RS", "Rio Grande do Sul", "South"), ("RO", "Rondônia", "North"),
    ("RR", "Roraima", "North"), ("SC", "Santa Catarina", "South"),
    ("SP", "São Paulo", "Southeast"), ("SE", "Sergipe", "Northeast"),
    ("TO", "Tocantins", "North"),
]


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def discover_batches(predictions_dir: Path) -> list[dict]:
    """Pair task files by their embedded batch ID and as-of date, never by mtime."""
    batches = {}
    with duckdb.connect() as connection:
        for task in REQUIRED:
            for path in sorted((predictions_dir / task).rglob("predictions.parquet")):
                columns = {row[0] for row in connection.execute(
                    "DESCRIBE SELECT * FROM read_parquet(?, hive_partitioning=false)", [str(path)]
                ).fetchall()}
                missing = REQUIRED[task] - columns
                if missing:
                    raise ValueError(f"{path}: missing columns {sorted(missing)}")
                identities = connection.execute("""
                    SELECT DISTINCT pipeline_run_id, CAST(as_of AS DATE) FROM read_parquet(?, hive_partitioning=false)
                """, [str(path)]).fetchall()
                if len(identities) != 1 or any(value is None for value in identities[0]):
                    raise ValueError(f"{path}: expected one nonempty batch ID / as-of date")
                run_id, as_of = identities[0]
                if not str(run_id).strip():
                    raise ValueError(f"{path}: empty batch run ID")
                key = (str(run_id), str(as_of))
                batch = batches.setdefault(key, {"pipeline_run_id": key[0], "as_of": key[1], "files": {}})
                if task in batch["files"]:
                    raise ValueError(f"Duplicate {task} output for {key}")
                batch["files"][task] = path.resolve()
    return list(batches.values())


def run_timestamp(run_id: str, files: dict) -> tuple[str, str]:
    """Legacy files lack execution timestamps; expose the timestamp's provenance."""
    with duckdb.connect() as connection:
        timestamps = []
        for path in files.values():
            columns = {row[0] for row in connection.execute(
                "DESCRIBE SELECT * FROM read_parquet(?, hive_partitioning=false)", [str(path)]).fetchall()}
            if "generated_at" in columns:
                value = connection.execute("SELECT CAST(min(generated_at) AS VARCHAR) FROM read_parquet(?, hive_partitioning=false)", [str(path)]).fetchone()[0]
                if value is not None:
                    timestamps.append(pd.Timestamp(value))
        if len(timestamps) == len(files) and all(value.tzinfo for value in timestamps):
            return min(timestamps).tz_convert("UTC").isoformat(), "prediction_generated_at"
    if "__" in run_id:
        try:
            timestamp = pd.Timestamp(run_id.split("__", 1)[1])
            if timestamp.tzinfo is not None:
                return timestamp.tz_convert("UTC").isoformat(), "airflow_run_id_logical_time"
        except (ValueError, TypeError):
            pass
    timestamp = max(path.stat().st_mtime for path in files.values())
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(), "prediction_file_mtime_fallback"


def load_references(connection, reference_dir: Path, as_of: str):
    """Take one-to-one dimensions from cleaned inputs; customer location is historical."""
    for name in ("products", "categories", "customers", "orders"):
        connection.execute(f"CREATE TABLE ref_{name} AS SELECT * FROM read_csv(?, header=true, all_varchar=true)",
                           [str(reference_dir / f"{name}.csv")])
    for table, key in (("products", "product_id"), ("categories", "product_category_name"),
                       ("customers", "customer_id"), ("orders", "order_id")):
        duplicates = connection.execute(
            f"SELECT count(*) FROM (SELECT {key} FROM ref_{table} GROUP BY {key} HAVING count(*) > 1)"
        ).fetchone()[0]
        if duplicates:
            raise ValueError(f"Duplicate reference keys in {table}.{key}")
    connection.execute("""
        CREATE TABLE dim_product AS
        SELECT p.product_id, p.product_category_name,
               coalesce(c.product_category_name_english, p.product_category_name, 'Unknown') AS category,
               try_cast(p.product_weight_g AS DOUBLE) AS product_weight_g,
               try_cast(p.product_length_cm AS DOUBLE) AS product_length_cm,
               try_cast(p.product_height_cm AS DOUBLE) AS product_height_cm,
               try_cast(p.product_width_cm AS DOUBLE) AS product_width_cm,
               try_cast(p.product_photos_qty AS INTEGER) AS product_photos_qty
        FROM ref_products p LEFT JOIN ref_categories c USING (product_category_name)
    """)
    connection.execute("""
        CREATE TABLE dim_customer AS
        SELECT c.customer_unique_id, c.customer_state, c.customer_city
        FROM ref_orders o JOIN ref_customers c USING (customer_id)
        WHERE try_cast(o.order_purchase_timestamp AS TIMESTAMP) < CAST(? AS TIMESTAMP)
        QUALIFY row_number() OVER (
            PARTITION BY c.customer_unique_id
            ORDER BY try_cast(o.order_purchase_timestamp AS TIMESTAMP) DESC, o.order_id DESC
        ) = 1
    """, [as_of])
    connection.execute("CREATE TABLE dim_state(state_code VARCHAR, state_name VARCHAR, region VARCHAR)")
    connection.executemany("INSERT INTO dim_state VALUES (?, ?, ?)", STATES)
    for name in ("products", "categories", "customers", "orders"):
        connection.execute(f"DROP TABLE ref_{name}")


def load_predictions(connection, batch: dict) -> dict:
    """Validate output grain and enrichment using temporary, non-materialized views."""
    counts = {}
    models = {}
    for task, path in batch["files"].items():
        connection.execute(f"CREATE TEMP VIEW raw_{task} AS SELECT * FROM read_parquet({sql_literal(str(path))}, hive_partitioning=false)")
        if connection.execute(
            f"SELECT count(*) FROM raw_{task} WHERE pipeline_run_id != ? OR CAST(as_of AS DATE) != CAST(? AS DATE)",
            [batch["pipeline_run_id"], batch["as_of"]],
        ).fetchone()[0]:
            raise ValueError(f"{task}: batch identity changed during discovery")
        count = connection.execute(f"SELECT count(*) FROM raw_{task}").fetchone()[0]
        counts[task] = count
        if connection.execute(f"SELECT count(*) FROM raw_{task} WHERE " + " OR ".join(
            f"{column} IS NULL" for column in PROVENANCE
        )).fetchone()[0]:
            raise ValueError(f"{task}: null provenance fields")
        identities = connection.execute(f"""
            SELECT DISTINCT model_run_id, model_name, model_role, training_pipeline_run_id FROM raw_{task}
        """).fetchall()
        if len(identities) != 1:
            raise ValueError(f"{task}: mixed model identities")
        models[task] = dict(zip(("model_run_id", "model_name", "model_role", "training_pipeline_run_id"), identities[0]))
    if connection.execute("""
        SELECT count(*) FROM raw_forecasting
        WHERE product_id IS NULL OR customer_state IS NULL OR prediction IS NULL
          OR NOT isfinite(prediction) OR prediction < 0
          OR purchase_date IS NULL OR target_date IS NULL
          OR CAST(purchase_date AS DATE) != CAST(as_of AS DATE)
          OR CAST(target_date AS DATE) != CAST(as_of AS DATE) + 1
    """).fetchone()[0]:
        raise ValueError("Invalid forecasting keys, dates or demand")
    if connection.execute("""
        SELECT count(*) FROM (
          SELECT product_id, customer_state, target_date FROM raw_forecasting
          GROUP BY ALL HAVING count(*) > 1
        )
    """).fetchone()[0]:
        raise ValueError("Duplicate forecasting product / state / target date")
    if connection.execute("""
        SELECT count(*) FROM raw_recommendation
        WHERE customer_unique_id IS NULL OR recommended_products IS NULL OR k IS NULL
           OR k < 1 OR len(recommended_products) > k
           OR len(list_distinct(recommended_products)) != len(recommended_products)
    """).fetchone()[0]:
        raise ValueError("Invalid recommendation customer, rank list or k")
    if connection.execute("""
        SELECT count(*) FROM (SELECT customer_unique_id FROM raw_recommendation GROUP BY ALL HAVING count(*) > 1)
    """).fetchone()[0]:
        raise ValueError("Duplicate recommendation customer")
    connection.execute("""
        CREATE TEMP VIEW forecasts AS
        SELECT f.*, coalesce(p.category, 'Unknown') AS category,
               p.product_weight_g, p.product_length_cm, p.product_height_cm, p.product_width_cm,
               p.product_photos_qty, p.product_id IS NOT NULL AS product_matched,
               coalesce(s.state_name, 'Unknown') AS state_name, coalesce(s.region, 'Unknown') AS region
        FROM raw_forecasting f LEFT JOIN dim_product p USING (product_id)
        LEFT JOIN dim_state s ON f.customer_state = s.state_code
    """)
    connection.execute("""
        CREATE TEMP VIEW recommendations AS
        WITH expanded AS (
            SELECT r.* EXCLUDE (recommended_products), u.product_id, u.rank
            FROM raw_recommendation r,
                 unnest(r.recommended_products) WITH ORDINALITY AS u(product_id, rank)
        )
        SELECT r.*,
               coalesce(p.category, 'Unknown') AS category, p.product_weight_g, p.product_photos_qty,
               p.product_id IS NOT NULL AS product_matched,
               coalesce(c.customer_state, 'Unknown') AS customer_state,
               coalesce(c.customer_city, 'Unknown') AS customer_city,
               coalesce(s.state_name, 'Unknown') AS state_name, coalesce(s.region, 'Unknown') AS region
        FROM expanded r
        LEFT JOIN dim_product p ON r.product_id = p.product_id
        LEFT JOIN dim_customer c ON r.customer_unique_id = c.customer_unique_id
        LEFT JOIN dim_state s ON c.customer_state = s.state_code
    """)
    connection.execute("""
        CREATE TEMP VIEW forecast_summary AS
        SELECT target_date, customer_state, state_name, region, category,
               sum(prediction) AS predicted_demand, count(*) AS product_state_rows,
               count(DISTINCT product_id) AS products
        FROM forecasts GROUP BY ALL
    """)
    connection.execute("""
        CREATE TEMP VIEW recommendation_summary AS
        SELECT customer_state, state_name, region, category, product_id,
               count(*) AS recommendation_count, avg(rank) AS average_rank
        FROM recommendations GROUP BY ALL
    """)
    counts["recommendation_items"] = connection.execute("SELECT count(*) FROM recommendations").fetchone()[0]
    counts["unmatched_forecast_products"] = connection.execute("SELECT count(*) FROM forecasts WHERE NOT product_matched").fetchone()[0]
    counts["unmatched_recommendation_products"] = connection.execute("SELECT count(*) FROM recommendations WHERE NOT product_matched").fetchone()[0]
    for task in REQUIRED:
        connection.execute(f"DROP VIEW raw_{task}")
    return {"row_counts": counts, "models": models}


def sql_literal(value):
    """Quote trusted file paths / IDs used in persistent view definitions."""
    return "'" + str(value).replace("'", "''") + "'"


def create_tables(connection):
    """Store prediction rows once per run; expose enrichment through views."""
    connection.execute("""
        CREATE TABLE IF NOT EXISTS prediction_runs (
            run_key VARCHAR PRIMARY KEY, pipeline_run_id VARCHAR, as_of DATE,
            run_timestamp VARCHAR, loaded_at VARCHAR, manifest JSON
        );
        CREATE TABLE IF NOT EXISTS forecast_predictions (
            run_key VARCHAR, as_of DATE, model_name VARCHAR, model_role VARCHAR,
            model_run_id VARCHAR, training_pipeline_run_id VARCHAR, pipeline_run_id VARCHAR,
            purchase_date DATE, product_id VARCHAR, customer_state VARCHAR, target_date DATE,
            prediction DOUBLE, generated_at VARCHAR,
            PRIMARY KEY (run_key, product_id, customer_state, target_date)
        );
        CREATE TABLE IF NOT EXISTS recommendation_predictions (
            run_key VARCHAR, as_of DATE, model_name VARCHAR, model_role VARCHAR,
            model_run_id VARCHAR, training_pipeline_run_id VARCHAR, pipeline_run_id VARCHAR,
            customer_unique_id VARCHAR, recommended_products VARCHAR[], k INTEGER, generated_at VARCHAR,
            PRIMARY KEY (run_key, customer_unique_id)
        );
    """)


def create_views(connection):
    """Enrich persisted predictions; views have no dependency on Parquet files."""
    connection.execute("""
        CREATE OR REPLACE VIEW forecasts AS
        SELECT f.*, coalesce(p.category, 'Unknown') AS category,
               p.product_weight_g, p.product_length_cm, p.product_height_cm, p.product_width_cm,
               p.product_photos_qty, p.product_id IS NOT NULL AS product_matched,
               coalesce(s.state_name, 'Unknown') AS state_name, coalesce(s.region, 'Unknown') AS region
        FROM forecast_predictions f JOIN prediction_runs b USING (run_key)
        LEFT JOIN dim_product p ON f.product_id = p.product_id
             AND p.reference_key = json_extract_string(b.manifest, '$.reference_key')
        LEFT JOIN dim_state s ON f.customer_state = s.state_code;

        CREATE OR REPLACE VIEW recommendations AS
        WITH expanded AS (
            SELECT r.* EXCLUDE (recommended_products), u.product_id, u.rank
            FROM recommendation_predictions r,
                 unnest(r.recommended_products) WITH ORDINALITY AS u(product_id, rank)
        )
        SELECT r.*, coalesce(p.category, 'Unknown') AS category, p.product_weight_g, p.product_photos_qty,
               p.product_id IS NOT NULL AS product_matched,
               coalesce(c.customer_state, 'Unknown') AS customer_state,
               coalesce(c.customer_city, 'Unknown') AS customer_city,
               coalesce(s.state_name, 'Unknown') AS state_name, coalesce(s.region, 'Unknown') AS region
        FROM expanded r JOIN prediction_runs b USING (run_key)
        LEFT JOIN dim_product p ON r.product_id = p.product_id
             AND p.reference_key = json_extract_string(b.manifest, '$.reference_key')
        LEFT JOIN dim_customer c ON r.customer_unique_id = c.customer_unique_id
             AND c.reference_key = json_extract_string(b.manifest, '$.reference_key')
             AND c.as_of = r.as_of
        LEFT JOIN dim_state s ON c.customer_state = s.state_code;

        CREATE OR REPLACE VIEW forecast_summary AS
        SELECT run_key, target_date, customer_state, state_name, region, category,
               sum(prediction) AS predicted_demand, count(*) AS product_state_rows,
               count(DISTINCT product_id) AS products FROM forecasts GROUP BY ALL;

        CREATE OR REPLACE VIEW recommendation_summary AS
        SELECT run_key, customer_state, state_name, region, category, product_id,
               count(*) AS recommendation_count, avg(rank) AS average_rank
        FROM recommendations GROUP BY ALL;
    """)


def store_dimensions(connection, validation, reference_key, as_of):
    schemas = {
        "product": "product_id VARCHAR, product_category_name VARCHAR, category VARCHAR, "
                   "product_weight_g DOUBLE, product_length_cm DOUBLE, product_height_cm DOUBLE, "
                   "product_width_cm DOUBLE, product_photos_qty INTEGER, reference_key VARCHAR",
        "customer": "customer_unique_id VARCHAR, customer_state VARCHAR, customer_city VARCHAR, "
                    "reference_key VARCHAR, as_of DATE",
        "state": "state_code VARCHAR, state_name VARCHAR, region VARCHAR",
    }
    for name, schema in schemas.items():
        connection.execute(f"CREATE TABLE IF NOT EXISTS dim_{name} ({schema})")
        frame = validation.execute(f"SELECT * FROM dim_{name}").df()
        if name != "state":
            frame["reference_key"] = reference_key
        if name == "customer":
            frame["as_of"] = pd.Timestamp(as_of).date()
        frame = frame.astype(object).where(pd.notna(frame), None)
        connection.register("dimension", frame)
        if name == "state":
            exists = connection.execute("SELECT count(*) FROM dim_state").fetchone()[0]
        elif name == "product":
            exists = connection.execute("SELECT count(*) FROM dim_product WHERE reference_key = ?", [reference_key]).fetchone()[0]
        else:
            exists = connection.execute("SELECT count(*) FROM dim_customer WHERE reference_key = ? AND as_of = CAST(? AS DATE)",
                                        [reference_key, as_of]).fetchone()[0]
        if not exists:
            connection.execute(f"INSERT INTO dim_{name} SELECT * FROM dimension")
        connection.unregister("dimension")


def load_batch(connection, batch, reference_dir, reference_hashes):
    run_key = identifier({"pipeline_run_id": batch["pipeline_run_id"], "as_of": batch["as_of"]})[:24]
    source_hashes = {task: file_hash(path) for task, path in batch["files"].items()}
    existing = connection.execute("SELECT manifest FROM prediction_runs WHERE run_key = ?", [run_key]).fetchone()
    if existing:
        if json.loads(existing[0])["prediction_hashes"] != source_hashes:
            raise ValueError(f"Prediction bytes changed for loaded run {batch['pipeline_run_id']}; use a new batch run ID")
        return {"run_key": run_key, "status": "already_loaded"}
    reference_key = identifier(reference_hashes)
    with duckdb.connect() as validation:
        load_references(validation, reference_dir, batch["as_of"])
        report = load_predictions(validation, batch)
        store_dimensions(connection, validation, reference_key, batch["as_of"])
    for task, table in (("forecasting", "forecast_predictions"), ("recommendation", "recommendation_predictions")):
        path = batch["files"][task]
        columns = {row[0] for row in connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?, hive_partitioning=false)", [str(path)]).fetchall()}
        fields = ", ".join(sorted(REQUIRED[task]))
        generated = "CAST(generated_at AS VARCHAR)" if "generated_at" in columns else "NULL::VARCHAR"
        connection.execute(
            f"INSERT INTO {table} BY NAME SELECT ? AS run_key, {fields}, {generated} AS generated_at "
            "FROM read_parquet(?, hive_partitioning=false)", [run_key, str(path)],
        )
    if source_hashes != {task: file_hash(path) for task, path in batch["files"].items()}:
        raise ValueError("Prediction files changed during loading")
    if reference_hashes != {name: file_hash(reference_dir / name) for name in REFERENCE_FILES}:
        raise ValueError("Reference files changed during loading")
    timestamp, timestamp_source = run_timestamp(batch["pipeline_run_id"], batch["files"])
    loaded_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": 1, "run_key": run_key, "pipeline_run_id": batch["pipeline_run_id"],
        "as_of": batch["as_of"], "run_timestamp": timestamp, "run_timestamp_source": timestamp_source,
        "loaded_at": loaded_at, "prediction_files": {task: str(path) for task, path in batch["files"].items()},
        "prediction_hashes": source_hashes, "reference_dir": str(reference_dir.resolve()),
        "reference_key": reference_key, "reference_hashes": reference_hashes, **report,
    }
    connection.execute("INSERT INTO prediction_runs VALUES (?, ?, CAST(? AS DATE), ?, ?, ?)",
                       [run_key, batch["pipeline_run_id"], batch["as_of"], timestamp, loaded_at, json.dumps(manifest)])
    return {"run_key": run_key, "status": "loaded", **report}


def run_pipeline(predictions_dir=ROOT / SETTINGS["inference"]["output_dir"],
                 reference_dir=ROOT / SETTINGS["data"]["processed_dir"],
                 database=PREDICTIONS_DATABASE, pipeline_run_id=None, batch_run_id=None):
    """Atomically load complete batches; retries do not duplicate rows."""
    predictions_dir, reference_dir, database = map(Path, (predictions_dir, reference_dir, database))
    database.parent.mkdir(parents=True, exist_ok=True)
    with locked():
        try:
            connection = duckdb.connect(str(database))
        except duckdb.IOException as error:
            raise RuntimeError(f"Cannot open {database} for loading. Close its DBeaver connection before running the loader. {error}") from error
        with connection:
            batches = discover_batches(predictions_dir)
            if batch_run_id:
                batches = [batch for batch in batches if batch["pipeline_run_id"] == batch_run_id]
            if not batches:
                raise ValueError(f"No prediction files found for the requested batch in {predictions_dir}")
            hashes = {name: file_hash(reference_dir / name) for name in REFERENCE_FILES}
            report = {"pipeline_run_id": pipeline_run_id, "loaded": [], "skipped_incomplete": []}
            connection.execute("BEGIN")
            try:
                create_tables(connection)
                for batch in batches:
                    if set(batch["files"]) != set(REQUIRED):
                        report["skipped_incomplete"].append({"pipeline_run_id": batch["pipeline_run_id"], "as_of": batch["as_of"]})
                        continue
                    report["loaded"].append(load_batch(connection, batch, reference_dir, hashes))
                if batch_run_id and report["skipped_incomplete"]:
                    raise ValueError("Both prediction tasks must complete before loading")
                if report["loaded"]:
                    create_views(connection)
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, default=ROOT / SETTINGS["inference"]["output_dir"])
    parser.add_argument("--reference-dir", type=Path, default=ROOT / SETTINGS["data"]["processed_dir"])
    parser.add_argument("--database", type=Path, default=PREDICTIONS_DATABASE)
    parser.add_argument("--pipeline-run-id")
    parser.add_argument("--batch-run-id", help="Load only this inference batch and require both task outputs")
    print(json.dumps(run_pipeline(**vars(parser.parse_args())), indent=2))
    return 0
