"""Versioned Olist ingestion; callable from Airflow or the local CLI."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import zipfile

import fsspec
import numpy as np
import pandas as pd
import pandera.pandas as pa

# Original Olist spelling, including `lenght`, is intentional.
TABLES = {
    'customers': ('olist_customers_dataset.csv', 'customer_id customer_unique_id customer_zip_code_prefix customer_city customer_state', ['customer_id']),
    'orders': ('olist_orders_dataset.csv', 'order_id customer_id order_status order_purchase_timestamp order_approved_at order_delivered_carrier_date order_delivered_customer_date order_estimated_delivery_date', ['order_id']),
    'order_items': ('olist_order_items_dataset.csv', 'order_id order_item_id product_id seller_id shipping_limit_date price freight_value', ['order_id', 'order_item_id']),
    'products': ('olist_products_dataset.csv', 'product_id product_category_name product_name_lenght product_description_lenght product_photos_qty product_weight_g product_length_cm product_height_cm product_width_cm', ['product_id']),
    'sellers': ('olist_sellers_dataset.csv', 'seller_id seller_zip_code_prefix seller_city seller_state', ['seller_id']),
    'payments': ('olist_order_payments_dataset.csv', 'order_id payment_sequential payment_type payment_installments payment_value', ['order_id', 'payment_sequential']),
    'reviews': ('olist_order_reviews_dataset.csv', 'review_id order_id review_score review_comment_title review_comment_message review_creation_date review_answer_timestamp', ['review_id', 'order_id']),
    'categories': ('product_category_name_translation.csv', 'product_category_name product_category_name_english', ['product_category_name']),
    'geolocation': ('olist_geolocation_dataset.csv', 'geolocation_zip_code_prefix geolocation_lat geolocation_lng geolocation_city geolocation_state', []),
}
NULLABLE = set('product_category_name product_name_lenght product_description_lenght product_photos_qty product_weight_g product_length_cm product_height_cm product_width_cm order_approved_at order_delivered_carrier_date order_delivered_customer_date review_comment_title review_comment_message'.split())
NUMERIC = set('order_item_id price freight_value product_name_lenght product_description_lenght product_photos_qty product_weight_g product_length_cm product_height_cm product_width_cm payment_sequential payment_installments payment_value review_score geolocation_lat geolocation_lng'.split())
TIMESTAMPS = set('order_purchase_timestamp order_approved_at order_delivered_carrier_date order_delivered_customer_date order_estimated_delivery_date shipping_limit_date review_creation_date review_answer_timestamp'.split())
INTEGERS = NUMERIC - {'price', 'freight_value', 'payment_value', 'geolocation_lat', 'geolocation_lng'}
STATUSES = 'delivered invoiced shipped processing unavailable canceled created approved'.split()
CONTRACT_VERSION = 'olist-ingestion-1.0.0'


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as out:
        json.dump(value, out, indent=2, sort_keys=True)
        temporary = out.name
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class Ingestion:
    def __init__(self, source='data/raw', storage='data/object_store', work='data/ingestion', warehouse='data/warehouse/olist.sqlite', purchase_date=None, run_id='local', max_rejected_fraction=0.01):
        if purchase_date:
            from datetime import date
            date.fromisoformat(purchase_date)
        if not 0 <= max_rejected_fraction <= 1:
            raise ValueError('max_rejected_fraction must be between 0 and 1')
        self.source = Path(source)
        self.storage = storage
        self.warehouse = Path(warehouse)
        self.purchase_date = purchase_date
        self.limit = max_rejected_fraction
        # Isolate Airflow runs; retries of a task use the same directory.
        self.work = Path(work) / hashlib.sha256(run_id.encode()).hexdigest()[:20]
        self.work.mkdir(parents=True, exist_ok=True)

    def stage(self, name):
        try:
            result = getattr(self, name)()
            atomic_json(self.work / f'{name}.status.json', {'status': 'success', 'result': result})
            return str(self.work)
        except Exception as exc:
            atomic_json(self.work / f'{name}.status.json', {'status': 'failed', 'error': str(exc)})
            raise

    def extract(self):
        dest = self.work / 'extracted'
        dest.mkdir(exist_ok=True)
        # Extract only known members; never unpack arbitrary archive paths.
        archive = zipfile.ZipFile(self.source) if self.source.is_file() else None
        try:
            for filename, _, _ in TABLES.values():
                content = archive.read(filename) if archive else (self.source / filename).read_bytes()
                (dest / filename).write_bytes(content)
        finally:
            if archive:
                archive.close()
        files = {filename: sha256(dest / filename) for filename, _, _ in TABLES.values()}
        version = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        atomic_json(self.work / 'manifest.json', {'dataset_version': version, 'contract_version': CONTRACT_VERSION, 'files': files})
        return {'dataset_version': version}

    def raw_storage(self):
        manifest = json.loads((self.work / 'manifest.json').read_text())
        fs, root = fsspec.core.url_to_fs(self.storage)
        dest = f'{root}/raw/dataset_version={manifest["dataset_version"]}'
        fs.makedirs(dest, exist_ok=True)
        for filename, expected in manifest['files'].items():
            target = f'{dest}/{filename}'
            content = (self.work / 'extracted' / filename).read_bytes()
            if fs.exists(target):
                with fs.open(target, 'rb') as stream:
                    actual = hashlib.sha256(stream.read()).hexdigest()
                if actual != expected:
                    raise ValueError(f'Immutable snapshot conflict: {target}')
            else:
                # Exclusive creation prevents replacing an existing object.
                with fs.open(target, 'xb') as stream:
                    stream.write(content)
        manifest_path = f'{dest}/manifest.json'
        if not fs.exists(manifest_path):
            with fs.open(manifest_path, 'xb') as stream:
                stream.write(json.dumps(manifest, sort_keys=True).encode())
        return {'snapshot_uri': dest}

    def _save(self, group, name, frame):
        directory = self.work / group
        directory.mkdir(exist_ok=True)
        frame.to_parquet(directory / f'{name}.parquet', index=False)

    def _load(self, group):
        return {name: pd.read_parquet(self.work / group / f'{name}.parquet') for name in TABLES}

    def validate(self):
        metrics = {}
        for name, (filename, columns, _) in TABLES.items():
            frame = pd.read_csv(self.work / 'extracted' / filename, dtype='string', encoding='utf-8-sig')
            expected = columns.split()
            schema = pa.DataFrameSchema({c: pa.Column(str, nullable=c in NULLABLE) for c in expected}, strict=False)
            # Row checks below preserve all reasons and source row numbers.
            schema.get_metadata()  # Ensure schema construction succeeds before handling rows.
            missing = set(expected) - set(frame.columns)
            if missing:
                raise ValueError(f'{filename}: missing columns {sorted(missing)}')
            extra = sorted(set(frame.columns) - set(expected))
            frame = frame[expected].copy()
            for col in frame:
                frame[col] = frame[col].str.strip().replace('', pd.NA)
            reasons = pd.Series('', index=frame.index, dtype='string')
            def flag(mask, message):
                nonlocal reasons
                mask = mask.fillna(False)
                reasons.loc[mask] = reasons.loc[mask] + message + ';'
            for col in expected:
                if col not in NULLABLE:
                    flag(frame[col].isna(), f'{col}:required')
                if col in NUMERIC:
                    parsed = pd.to_numeric(frame[col], errors='coerce')
                    flag(frame[col].notna() & (parsed.isna() | ~np.isfinite(parsed.astype(float))), f'{col}:numeric')
                    if col in INTEGERS:
                        flag(parsed.notna() & (parsed % 1 != 0), f'{col}:integer')
                    minimum = 1 if col in {'order_item_id', 'payment_sequential', 'review_score'} else 0
                    if col not in {'geolocation_lat', 'geolocation_lng'}:
                        flag(parsed < minimum, f'{col}:minimum')
                    if col == 'review_score':
                        flag(parsed > 5, f'{col}:maximum')
                    if col.startswith('geolocation_'):
                        bound = 90 if col.endswith('lat') else 180
                        flag(parsed.abs() > bound, f'{col}:coordinate')
                if col in TIMESTAMPS:
                    parsed = pd.to_datetime(frame[col], format='%Y-%m-%d %H:%M:%S', errors='coerce')
                    flag(frame[col].notna() & parsed.isna(), f'{col}:timestamp')
                if col.endswith('_state'):
                    flag(~frame[col].isin('AC AL AP AM BA CE DF ES GO MA MT MS MG PA PB PR PE PI RJ RN RS RO RR SC SP SE TO'.split()), f'{col}:state')
            if name == 'orders':
                flag(~frame.order_status.isin(STATUSES), 'order_status:unknown')
            rejected = reasons.ne('')
            quarantine = frame.loc[rejected].copy()
            quarantine['source_row_number'] = quarantine.index + 2
            quarantine['failure_reason'] = reasons[rejected]
            self._quarantine(name, quarantine, filename)
            accepted = frame.loc[~rejected].copy()
            # Pandera enforces structural/null rules after collecting invalid rows.
            schema.validate(accepted, lazy=True)
            self._save('validated', name, accepted)
            metrics[name] = {'input_rows': len(frame), 'invalid_rows': int(rejected.sum()), 'extra_columns': extra}
        atomic_json(self.work / 'validation_metrics.json', metrics)
        return metrics

    def _quarantine(self, name, frame, filename, suffix='invalid'):
        frame = frame.copy()
        frame['source_file'] = filename
        frame['dataset_version'] = json.loads((self.work / 'manifest.json').read_text())['dataset_version']
        frame['contract_version'] = CONTRACT_VERSION
        self._save('quarantine', f'{name}_{suffix}', frame)

    def clean(self):
        tables = self._load('validated')
        counts = {}
        for name, frame in tables.items():
            filename, _, keys = TABLES[name]
            exact = frame.duplicated()
            duplicates = frame.loc[exact].copy()
            duplicates['failure_reason'] = 'exact_duplicate'
            duplicates['source_row_number'] = duplicates.index + 2
            self._quarantine(name, duplicates, filename, 'duplicates')
            frame = frame.loc[~exact].copy()
            if keys and frame.duplicated(keys, keep=False).any():
                conflicts = frame.loc[frame.duplicated(keys, keep=False)].copy()
                conflicts['failure_reason'] = 'conflicting_primary_key'
                self._quarantine(name, conflicts, filename, 'conflicts')
                raise ValueError(f'{name}: conflicting duplicate keys')
            for col in frame:
                if col in NUMERIC:
                    frame[col] = pd.to_numeric(frame[col])
                    if col in INTEGERS:
                        frame[col] = frame[col].astype('Int64')
                elif col in TIMESTAMPS:
                    frame[col] = pd.to_datetime(frame[col], format='%Y-%m-%d %H:%M:%S')
                elif col.endswith('_city') or col.startswith('product_category_name'):
                    frame[col] = frame[col].str.lower()
                elif col.endswith('_zip_code_prefix'):
                    frame[col] = frame[col].str.zfill(5)
            counts[name] = {'exact_duplicates_removed': int(exact.sum()), 'missing_references': 0}
            tables[name] = frame
        # Parent-first processing ensures rejected parents cannot leave orphan children.
        relations = [('orders', 'customer_id', 'customers'), ('order_items', 'order_id', 'orders'), ('order_items', 'product_id', 'products'), ('order_items', 'seller_id', 'sellers'), ('payments', 'order_id', 'orders'), ('reviews', 'order_id', 'orders')]
        for child, key, parent in relations:
            frame = tables[child]
            invalid = ~frame[key].isin(tables[parent][key])
            bad = frame.loc[invalid].copy()
            bad['failure_reason'] = f'{key}:missing_{parent}'
            bad['source_row_number'] = bad.index + 2
            self._quarantine(child, bad, TABLES[child][0], f'orphan_{key}')
            counts[child]['missing_references'] += int(invalid.sum())
            tables[child] = frame.loc[~invalid].copy()
        metrics = json.loads((self.work / 'validation_metrics.json').read_text())
        violations = []
        for name, frame in tables.items():
            self._save('clean', name, frame)
            rejected = metrics[name]['invalid_rows'] + counts[name]['missing_references']
            if rejected / max(1, metrics[name]['input_rows']) > self.limit:
                violations.append(name)
        atomic_json(self.work / 'clean_metrics.json', counts)
        if violations:
            raise ValueError(f'Rejected-row threshold {self.limit} exceeded: {violations}')
        return counts

    def feature_tables(self):
        t = self._load('clean')
        payments = t['payments'].groupby('order_id', as_index=False).agg(payment_value_total=('payment_value', 'sum'), payment_count=('payment_sequential', 'count'))
        # Reviews are descriptive outcomes; never pre-purchase model features.
        reviews = t['reviews'].groupby('order_id', as_index=False).agg(review_score_mean=('review_score', 'mean'), review_count=('review_id', 'count'))
        orders = t['orders'].merge(t['customers'], on='customer_id', validate='many_to_one')
        orders = orders.merge(payments, on='order_id', how='left', validate='one_to_one').merge(reviews, on='order_id', how='left', validate='one_to_one')
        orders['purchase_date'] = orders.order_purchase_timestamp.dt.strftime('%Y-%m-%d')
        if self.purchase_date:
            orders = orders.loc[orders.purchase_date.eq(self.purchase_date)]
        products = t['products'].merge(t['categories'], on='product_category_name', how='left', validate='many_to_one')
        products['product_category_name_english'] = products.product_category_name_english.fillna(products.product_category_name).fillna('unknown')
        items = t['order_items'].merge(orders, on='order_id', validate='many_to_one')
        items = items.merge(products, on='product_id', validate='many_to_one').merge(t['sellers'], on='seller_id', validate='many_to_one')
        interactions = items.loc[items.order_status.eq('delivered'), ['order_id', 'order_item_id', 'customer_unique_id', 'product_id', 'order_purchase_timestamp', 'order_delivered_customer_date', 'purchase_date']].copy()
        interactions['interaction_type'] = 'purchase'
        # Placed order-item counts, all source statuses; no inference of inventory.
        demand = items.groupby(['purchase_date', 'product_id', 'customer_state'], as_index=False).size().rename(columns={'size': 'order_item_count'})
        for name, frame in {'orders': orders, 'order_items': items, 'purchase_interactions': interactions, 'daily_demand': demand}.items():
            self._save('curated', name, frame)
        return {'orders': len(orders), 'order_items': len(items), 'interactions': len(interactions), 'daily_demand': len(demand)}

    def publish(self):
        self.warehouse.parent.mkdir(parents=True, exist_ok=True)
        manifest = json.loads((self.work / 'manifest.json').read_text())
        names = ['orders', 'order_items', 'purchase_interactions', 'daily_demand']
        frames = {n: pd.read_parquet(self.work / 'curated' / f'{n}.parquet') for n in names}
        # All table/partition replacements commit together. No append-on-retry.
        with sqlite3.connect(self.warehouse) as connection:
            connection.execute('BEGIN IMMEDIATE')
            try:
                for name, frame in frames.items():
                    frame['dataset_version'] = manifest['dataset_version']
                    frame['contract_version'] = CONTRACT_VERSION
                    for col in frame.select_dtypes(include=['datetime']):
                        frame[col] = frame[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                    columns = list(frame.columns)
                    def q(value):
                        return '"' + value.replace('"', '""') + '"'
                    definitions = ', '.join(f'{q(c)} {"REAL" if pd.api.types.is_numeric_dtype(frame[c]) else "TEXT"}' for c in columns)
                    connection.execute(f'CREATE TABLE IF NOT EXISTS {q(name)} ({definitions})')
                    existing = [r[1] for r in connection.execute(f'PRAGMA table_info({q(name)})')]
                    if existing != columns:
                        raise ValueError(f'Warehouse schema mismatch: {name}')
                    if self.purchase_date:
                        connection.execute(f'DELETE FROM {q(name)} WHERE purchase_date = ?', (self.purchase_date,))
                    else:
                        connection.execute(f'DELETE FROM {q(name)}')
                    placeholders = ','.join('?' for _ in columns)
                    values = frame.astype(object).where(pd.notna(frame), None)
                    connection.executemany(f'INSERT INTO {q(name)} VALUES ({placeholders})', values.itertuples(index=False, name=None))
                    connection.execute(f'CREATE INDEX IF NOT EXISTS {q(name + "_purchase_date")} ON {q(name)} (purchase_date)')
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return {'warehouse': str(self.warehouse), 'purchase_date': self.purchase_date, 'tables': {n: len(f) for n, f in frames.items()}}

    def quality_report(self):
        stages = ['extract', 'raw_storage', 'validate', 'clean', 'feature_tables', 'publish']
        statuses = {n: json.loads((self.work / f'{n}.status.json').read_text()) if (self.work / f'{n}.status.json').exists() else {'status': 'not_run'} for n in stages}
        report = {'status': 'success' if all(s['status'] == 'success' for s in statuses.values()) else 'failed', 'purchase_date': self.purchase_date, 'contract_version': CONTRACT_VERSION, 'max_rejected_fraction': self.limit, 'stages': statuses}
        for filename in ['manifest', 'validation_metrics', 'clean_metrics']:
            path = self.work / f'{filename}.json'
            if path.exists():
                report[filename] = json.loads(path.read_text())
        atomic_json(self.work / 'quality_report.json', report)
        print(json.dumps({'status': report['status'], 'report': str(self.work / 'quality_report.json')}))
        if report['status'] != 'success':
            raise RuntimeError(f'Ingestion failed; see {self.work / "quality_report.json"}')
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='data/raw', help='Extracted CSV directory or Olist ZIP')
    parser.add_argument('--storage', default='data/object_store', help='Filesystem path or fsspec object-storage URI')
    parser.add_argument('--work', default='data/ingestion')
    parser.add_argument('--warehouse', default='data/warehouse/olist.sqlite')
    parser.add_argument('--purchase-date')
    parser.add_argument('--run-id', default='local')
    parser.add_argument('--max-rejected-fraction', type=float, default=0.01)
    job = Ingestion(**vars(parser.parse_args()))
    # Remove only status markers so a rerun cannot report stale stage success.
    for path in job.work.glob('*.status.json'):
        path.unlink()
    try:
        for stage in ['extract', 'raw_storage', 'validate', 'clean', 'feature_tables', 'publish']:
            job.stage(stage)
    finally:
        job.quality_report()


if __name__ == '__main__':
    main()
