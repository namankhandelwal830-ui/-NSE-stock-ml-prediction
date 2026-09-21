"""
support_resistance.py
----------------------
Detects support & resistance price levels from swing highs/lows in the
price history, then clusters nearby levels together (since a "level" is
rarely one exact price — it's a zone that price has bounced off of
multiple times).
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


def find_swing_points(df: pd.DataFrame, order: int = 5):
    """
    Finds local swing highs (potential resistance) and swing lows
    (potential support) using a simple "higher/lower than N neighbours
    on each side" rule.

    order : how many candles on each side must be lower/higher for a
            point to count as a swing high/low. Bigger order = fewer,
            more significant swing points.
    """
    highs = df["High"].values
    lows = df["Low"].values

    high_idx = argrelextrema(highs, np.greater, order=order)[0]
    low_idx = argrelextrema(lows, np.less, order=order)[0]

    swing_highs = df.iloc[high_idx]["High"]
    swing_lows = df.iloc[low_idx]["Low"]
    return swing_highs, swing_lows


def cluster_levels(prices: pd.Series, tolerance_pct: float = 1.5):
    """
    Groups nearby price levels together (within tolerance_pct of each
    other) into a single level, since price rarely touches the exact
    same value twice. Returns a list of dicts: {price, touches}.
    """
    if prices.empty:
        return []

    sorted_prices = sorted(prices.tolist())
    clusters = [[sorted_prices[0]]]

    for p in sorted_prices[1:]:
        if abs(p - clusters[-1][-1]) / clusters[-1][-1] * 100 <= tolerance_pct:
            clusters[-1].append(p)
        else:
            clusters.append([p])

    return [{"price": float(np.mean(c)), "touches": len(c)} for c in clusters]


def detect_support_resistance(df: pd.DataFrame, order: int = 5,
                               tolerance_pct: float = 1.5, max_levels: int = 4) -> dict:
    """
    Full pipeline: find swing points -> cluster into levels -> keep only
    the most-touched (most significant) levels, ranked Major/Minor.

    Returns {"resistance": [...], "support": [...]}, each a list of
    dicts: {"price": float, "touches": int, "strength": "Major"/"Minor"}
    """
    swing_highs, swing_lows = find_swing_points(df, order=order)

    resistance = cluster_levels(swing_highs, tolerance_pct)
    support = cluster_levels(swing_lows, tolerance_pct)

    # Most-touched levels are the most significant ones — keep top N
    resistance = sorted(resistance, key=lambda x: -x["touches"])[:max_levels]
    support = sorted(support, key=lambda x: -x["touches"])[:max_levels]

    for i, r in enumerate(resistance):
        r["strength"] = "Major" if i < max(1, len(resistance) // 2) else "Minor"
    for i, s in enumerate(support):
        s["strength"] = "Major" if i < max(1, len(support) // 2) else "Minor"

    # Sort by price for a cleaner display (high to low)
    resistance = sorted(resistance, key=lambda x: -x["price"])
    support = sorted(support, key=lambda x: -x["price"])

    return {"resistance": resistance, "support": support}
