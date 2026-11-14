# Purchase-based recommendation baseline

Implementation: `scripts/features/build_recommendation_features.py`.

## 1. Prepare the warehouse

Run the existing cleaning and curated-data scripts if the warehouse does not exist:

```bash
.venv/bin/python scripts/data_processing/clean_olist_data.py
.venv/bin/python scripts/data_processing/prepare_curated_data.py
```

The recommender reads `data/warehouse/olist.duckdb`, table `order_items`, in read-only mode.
Use `customer_unique_id` to identify a person across orders; `customer_id` is order-specific in Olist.

## 2. Choose a customer, region and cutoff

```bash
.venv/bin/python scripts/features/build_recommendation_features.py \
  --customer-id 871766c5855e863f6eccc05f988b23cb \
  --state RJ \
  --as-of 2018-08-01 \
  --candidate-limit 50 \
  --top-k 10
```

Only purchases made AND delivered strictly before the cutoff are included. This avoids
using eventual delivery status before delivery was known. Popularity counts use purchase
dates within the preceding 90 days among those known delivered purchases.

## 3. Generate candidates

The script computes global and regional purchase counts, customer category preferences,
and shared-order counts. Each source contributes up to `candidate-limit` products:

- Global popularity, including a fallback to all-time counts when recent counts are empty.
- Regional popularity.
- Popular products in categories the customer has purchased (a basic similarity proxy).
- Products in historical baskets containing any of the customer's purchased products.

The union is deduplicated. Previously purchased products are excluded before source
selection by default. Use `--include-purchased` for repeat-purchase recommendations.
There are at most four times `candidate-limit` candidates, not a customer-by-catalog cross join.

## 4. Build one row per customer–candidate pair

Features include category affinity (fraction of the customer's purchased item rows in
that category), regional/global popularity, co-purchase basket count, historical median
price, customer segment, recent purchases of that product, and time since its last purchase.
Co-purchase counts count each order once per candidate, regardless of item quantity.
Default purchased-item exclusion makes customer–candidate purchase counts zero and
last-purchase recency missing; these fields become useful with `--include-purchased`.
Price is a historical summary, not a current selling price.

## 5. Score and rank

Popularity and co-purchase counts are divided by their respective maxima across the
eligible catalog. Category affinity is already between zero and one.

```text
score = 0.4 × category_affinity
      + 0.3 × regional_popularity_score
      + 0.2 × global_popularity_score
      + 0.1 × co_purchase_score
```

For customers with no history:

```text
score = 0.6 × regional_popularity_score + 0.4 × global_popularity_score
```

Sort descending by score, using product ID to break ties. These heuristic scores are
not purchase probabilities; weights have not been optimized on a holdout set.

## 6. Inspect outputs

Default directory: `data/features/recommendations/`.

- `candidates.parquet`: candidate features, source names, scores and ranks.
- `recommendations.parquet`: top K candidate rows.
- `metadata.json`: cutoff, customer, candidate count, weights and data limitations.

Each invocation overwrites these filenames. Use separate `--output-dir` paths to retain
multiple customers or snapshots. An exhausted eligible catalog yields empty outputs.

## 7. Add missing sources and evaluate before model training

Olist has no views, brands, inventory snapshots, recalls or promotions. This script
does not invent those features or claim products are currently available. A live
catalog/eligibility source must be added before offering these products for purchase.

For an offline evaluation, generate candidates at historical cutoffs and use purchases
in the following period as held-out relevance labels. Measure candidate recall and
Recall@K/NDCG@K; never use that future period to calculate the snapshot's features.
Repeat across cutoffs before selecting weights or training a ranker. Keep this baseline
as an independently callable fallback. No ranking model or API integration is included.

## Verification

```bash
.venv/bin/python -m unittest discover -s tests -p test_recommendation_features.py
```

Tests cover exclusion and category/co-purchase ranking, new-customer all-time fallback,
and exclusion of purchases whose delivery occurs at or after the cutoff.
