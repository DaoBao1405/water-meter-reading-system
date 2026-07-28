from __future__ import annotations

import numpy as np
import pandas as pd


def clean_pressure_data(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["NGAYGIO"] = pd.to_datetime(result["NGAYGIO"])
    result["APLUC"] = pd.to_numeric(result["APLUC"], errors="coerce")
    result = result.sort_values("NGAYGIO").drop_duplicates("NGAYGIO").reset_index(drop=True)
    result["APLUC"] = result["APLUC"].interpolate(limit_direction="both")
    return result


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    minute_of_day = result["NGAYGIO"].dt.hour * 60 + result["NGAYGIO"].dt.minute
    result["sin_day"] = np.sin(2 * np.pi * minute_of_day / 1440)
    result["cos_day"] = np.cos(2 * np.pi * minute_of_day / 1440)

    day_of_week = result["NGAYGIO"].dt.dayofweek
    result["sin_week"] = np.sin(2 * np.pi * day_of_week / 7)
    result["cos_week"] = np.cos(2 * np.pi * day_of_week / 7)
    return result


def make_windows(features: np.ndarray, target: np.ndarray, timestamps: np.ndarray, seq_len: int):
    xs, ys, ts = [], [], []
    for i in range(seq_len, len(features)):
        xs.append(features[i - seq_len : i])
        ys.append(target[i])
        ts.append(timestamps[i])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), np.asarray(ts)


def chronological_split(n: int, train_ratio: float = 0.70, val_ratio: float = 0.15):
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n)
