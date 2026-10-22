from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class SplitFrames:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def demand_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    daily = (
        df.groupby(df["order_purchase_timestamp"].dt.date)
        .agg(demand=("quantity", "sum"), avg_price=("price", "mean"))
        .reset_index(names=["date"])
        .sort_values("date")
    )
    daily["day_index"] = range(len(daily))
    daily["lag_1"] = daily["demand"].shift(1).fillna(daily["demand"].median())
    return daily


def time_based_split(frame: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15) -> SplitFrames:
    ordered = frame.sort_values("date").reset_index(drop=True)
    n_rows = len(ordered)
    if n_rows < 3:
        raise ValueError("At least 3 rows are required for time-based train/validation/test splits")
    train_end = max(1, int(n_rows * train_ratio))
    val_size = max(1, int(n_rows * val_ratio))
    val_end = min(n_rows - 1, train_end + val_size)
    if val_end <= train_end:
        val_end = min(n_rows - 1, train_end + 1)
    return SplitFrames(
        train=ordered.iloc[:train_end],
        validation=ordered.iloc[train_end:val_end],
        test=ordered.iloc[val_end:],
    )
