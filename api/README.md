# Inference API

FastAPI exposes the activated forecasting and recommendation models through HTTP.
It computes predictions on demand; it does not return saved batch predictions,
train/activate models, or insert API responses into the predictions DuckDB database.

## Start the server

From the project root, with dependencies installed and models activated:

```bash
source .venv/bin/activate
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000 --workers 1
```

- Swagger UI, including **Try it out**: <http://127.0.0.1:8000/docs>
- Live OpenAPI contract: <http://127.0.0.1:8000/openapi.json>
- ReDoc: <http://127.0.0.1:8000/redoc>
- Exported contracts: [YAML](../documentation/inference-openapi.yaml) and
  [JSON](../documentation/inference-openapi.json)

FastAPI, Uvicorn and Pydantic are already pinned in the project requirements.
The server binds to localhost in this example. Authentication and public hosting
are not configured.

## Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | HTTP liveness; does not require available models/data |
| GET | `/v1/models` | Load and inspect both active models; returns 503 if unavailable |
| POST | `/v1/inference/forecast` | Predict next-day demand for up to 100 product/state pairs |
| POST | `/v1/inference/recommendations` | Rank products for one customer using past delivered history |

### Forecast demand

```bash
curl -X POST http://127.0.0.1:8000/v1/inference/forecast \
  -H 'Content-Type: application/json' \
  -d '{"as_of":"2018-08-01","items":[{"product_id":"0c44cf18fca971245b41119ce72861c0","customer_state":"SP"}]}'
```

`as_of` is the prepared feature date. The result predicts `as_of + 1 day` and
returns the product ID, state, target date and demand estimate. Results preserve
request order. Product/state pairs must be unique. If any pair lacks features,
the complete request returns 404 rather than silently returning a partial result.

Inputs come from `inference.forecast_features` in `configs/settings.yaml`. Sending
a date does not generate new features; prepare data covering that date first.
Only known Brazilian state codes are accepted. Raw feature vectors and file paths
are not accepted from HTTP clients.

### Recommend products

```bash
curl -X POST http://127.0.0.1:8000/v1/inference/recommendations \
  -H 'Content-Type: application/json' \
  -d '{"as_of":"2018-08-01","customer_unique_id":"0000366f3b9a7992bf8c76cfdf3221e2","k":10}'
```

The service reads `inference.warehouse` and uses only purchases and deliveries
before `as_of`, matching batch inference. `k` defaults to the active model's
configured value and must not exceed its candidate limit or the API limit of 100.
The response includes one-based ranks and product IDs, not prediction scores.
Fewer than k products may be returned if the eligible catalog is smaller.
Unknown customers set `cold_start=true` and use the active model's fallback
behavior. A date with no eligible history returns 404.

Every successful inference response includes a request ID, generation timestamp,
as-of date, model name/role, MLflow model run ID, and training pipeline run ID.

### Errors

```json
{
  "error": {
    "code": "features_not_found",
    "message": "No prepared features for the requested date/product/state.",
    "details": ["missing-product/SP"]
  }
}
```

- **404:** requested forecast features or recommendation history absent.
- **422:** invalid date/state/ID, duplicate forecast pairs, request size, or k.
- **503:** active models unavailable/unapproved, invalid inputs/model output,
  a conflicting workspace reservation, or an unavailable/locked warehouse.

Disconnect DBeaver from the **Olist source warehouse** if recommendation requests
report a database lock. API inference does not read `predictions.duckdb`.

## Model consistency and concurrency

Each request acquires the shared workspace guard, loads the current approved
bundle, reads the data and predicts under that guard. Successful activation is
visible on the next request without restarting the API. Models are not cached
across requests. Requests are serialized by the local filesystem guard and are
rejected with 503 while another workflow holds a persistent workspace reservation.

This is intended for the existing local, single-host workflow. Recommendation
snapshots are rebuilt from history on each request, which can be expensive.
There is no distributed serving layer or shared feature cache.

## Export and test

```bash
python scripts/api/export_openapi.py
python -m unittest discover -s tests -p 'test_inference_api.py'
```

The contract is generated from route/request/response definitions. The tests
check it matches the application, and exercise real serialized baseline models,
validation, rank order, cold starts, missing inputs, workspace conflicts and
activation changes.
