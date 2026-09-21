"""
preprocessing.py
-----------------
Cleans raw OHLCV data before feature engineering:
  1. Missing value handling
  2. Outlier detection & treatment
  3. Feature normalisation / scaling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def handle_missing_values(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    """
    Handles missing values in OHLCV data.

    method options:
      - 'ffill'   : forward-fill (carry last known price forward) — most common for price data
      - 'bfill'   : backward-fill
      - 'interpolate' : linear interpolation between known points
      - 'drop'    : drop rows with any NaN
    """
    df = df.copy()

    # Report how much is missing before we touch it
    missing_count = df.isna().sum().sum()
    if missing_count > 0:
        print(f"[INFO] Found {missing_count} missing values. Handling with '{method}'.")

    if method == "ffill":
        df = df.ffill().bfill()  # bfill at the very start in case first row is NaN
    elif method == "bfill":
        df = df.bfill().ffill()
    elif method == "interpolate":
        df = df.interpolate(method="linear").ffill().bfill()
    elif method == "drop":
        df = df.dropna()
    else:
        raise ValueError(f"Unknown method: {method}")

    return df


def detect_outliers_iqr(df: pd.DataFrame, columns: list = None, factor: float = 1.5) -> pd.DataFrame:
    """
    Flags outliers using the IQR (Interquartile Range) method.
    Adds a boolean column '<col>_outlier' for each numeric column checked.

    A value is an outlier if it falls outside:
        [Q1 - factor*IQR, Q3 + factor*IQR]
    """
    df = df.copy()
    if columns is None:
        columns = ["Open", "High", "Low", "Close", "Volume"]

    for col in columns:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        df[f"{col}_outlier"] = (df[col] < lower) | (df[col] > upper)

    return df


def treat_outliers(df: pd.DataFrame, columns: list = None, factor: float = 1.5, method: str = "cap") -> pd.DataFrame:
    """
    Treats outliers after detection.

    method options:
      - 'cap'    : clip values to the IQR bounds (winsorization) — preserves row count, recommended for price series
      - 'remove' : drop rows flagged as outliers
    """
    df = df.copy()
    if columns is None:
        columns = ["Open", "High", "Low", "Close", "Volume"]

    for col in columns:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR

        if method == "cap":
            df[col] = df[col].clip(lower=lower, upper=upper)
        elif method == "remove":
            df = df[(df[col] >= lower) & (df[col] <= upper)]
        else:
            raise ValueError(f"Unknown method: {method}")

    # drop any leftover outlier flag columns from detect_outliers_iqr if present
    df = df[[c for c in df.columns if not c.endswith("_outlier")]]
    return df


def normalise_features(df: pd.DataFrame, columns: list, method: str = "standard"):
    """
    Scales numeric features. Returns (scaled_df, fitted_scaler) so the SAME
    scaler can later be applied to test/live data (avoids data leakage).

    method options:
      - 'standard' : zero mean, unit variance (StandardScaler) — good default for SVM/linear models
      - 'minmax'   : scales to [0, 1] range — good when you want bounded features
    """
    df = df.copy()
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()

    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler


def clean_pipeline(raw_df: pd.DataFrame, missing_method: str = "ffill",
                    outlier_method: str = "cap") -> pd.DataFrame:
    """
    Convenience wrapper that runs the full cleaning pipeline in order:
    missing values -> outlier treatment.
    Normalisation is deliberately NOT included here — it's applied later,
    only on the feature columns, after feature engineering (see features.py),
    so that raw OHLCV values used for plotting/backtesting stay in original units.
    """
    df = handle_missing_values(raw_df, method=missing_method)
    df = treat_outliers(df, method=outlier_method)
    return df
