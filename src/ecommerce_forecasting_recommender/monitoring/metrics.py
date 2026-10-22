from __future__ import annotations

from collections.abc import Iterator
from time import perf_counter
from typing import Self

from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter("api_requests_total", "Total API requests", ["endpoint"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "API request latency", ["endpoint"])


class LatencyTracker:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self.start = 0.0

    def __enter__(self) -> Self:
        REQUEST_COUNT.labels(endpoint=self.endpoint).inc()
        self.start = perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        REQUEST_LATENCY.labels(endpoint=self.endpoint).observe(perf_counter() - self.start)


def latency(endpoint: str) -> Iterator[LatencyTracker]:
    return LatencyTracker(endpoint)
