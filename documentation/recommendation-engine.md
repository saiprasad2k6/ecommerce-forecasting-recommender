# Recommendation Engine

## Overview

The recommendation engine creates purchase-based product recommendations from the Olist DuckDB warehouse.

It uses delivered purchases known before the requested cutoff date. This prevents future purchases from influencing recommendations.

## Data Source

The engine reads:

```text
data/warehouse/olist.duckdb
```

It uses delivered records from the `order_items` table, including:

- `customer_unique_id`
- `product_id`
- `customer_state`
- `product_category_name_english`
- `order_purchase_timestamp`
- `order_delivered_customer_date`

## Candidate Generation

Products are collected from four sources:

1. Globally popular products purchased during the last 90 days
2. Regionally popular products purchased during the last 90 days
3. Products matching the customer's recent categories
4. Products frequently purchased in the same orders as the customer's products

The candidate lists are merged and deduplicated. Products already purchased by the customer are excluded by default. Use `--include-purchased` to include them.

## Ranking

Each candidate receives normalized scores:

| Feature | Weight |
| --- | ---: |
| Category affinity | 0.40 |
| Regional popularity | 0.30 |
| Global popularity | 0.20 |
| Co-purchase score | 0.10 |

The baseline score is:

```text
baseline_score =
    category_affinity * 0.40
  + regional_popularity * 0.30
  + global_popularity * 0.20
  + co_purchase_score * 0.10
```

Recommendations are sorted by descending score and assigned a rank.

## New Customers

Customers without purchase history use a fallback score:

```text
regional popularity * 0.60
+ global popularity * 0.40
```

This provides recommendations when personalized purchase history is unavailable.

## Usage

```bash
.venv/bin/python scripts/features/build_recommendation_features.py \
  --customer-id 871766c5855e863f6eccc05f988b23cb \
  --state RJ \
  --as-of 2018-08-01 \
  --top-k 10
```

The `--as-of` date is exclusive. Purchases on or after that date are not used.

## Outputs

Results are written to:

```text
data/features/recommendations/
├── candidates.parquet
├── recommendations.parquet
└── metadata.json
```

## Limitations

The Olist dataset does not contain product views, clicks, inventory availability, stockout information, promotions, brands, discounts, or recall status. These signals are not inferred and should be added only when reliable source data becomes available.