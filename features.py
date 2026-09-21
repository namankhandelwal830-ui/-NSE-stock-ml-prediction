"""
features.py
------------
Feature engineering — a focused, trader-style indicator set:
  - EMA 9 / EMA 21   (fast trend / crossover signal)
  - SMA 200          (long-term trend filter)
  - RSI 14           (momentum / overbought-oversold)
  - VWAP (rolling)   (volume-weighted fair price)
  - Price action     (candle body %, gap, short-term volatility)

Plus the prediction targets used by models.py.
"""

import pandas as pd
import numpy as np


def add_ema(df, window):
    """Exponential Moving Average — weights recent prices more than older ones."""
    df[f"EMA_{window}"] = df["Close"].ewm(span=window, adjust=False).mean()
    return df


def add_sma(df, window):
    """Simple Moving Average — used here as a long-term (200-day) trend filter."""
    df[f"SMA_{window}"] = df["Close"].rolling(window=window, min_periods=window).mean()
    return df


def add_rsi(df, window=14):
    """
    Relative Strength Index — momentum oscillator (0-100).
    >70 typically = overbought, <30 = oversold.
    """
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df[f"RSI_{window}"] = 100 - (100 / (1 + rs))
    return df


def add_vwap(df, window=20):
    """
    Rolling VWAP (Volume Weighted Average Price) over `window` days.
    True intraday VWAP resets every session; for daily bars we use a rolling
    window as a practical "fair value" reference line traders commonly watch.
    """
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    pv = typical_price * df["Volume"]
    df[f"VWAP_{window}"] = pv.rolling(window=window).sum() / df["Volume"].rolling(window=window).sum()
    return df


def add_price_action_features(df):
    """
    Simple price-action-derived numeric features (not indicators from a
    library — just describing the shape of each candle and recent moves):
      - Candle_Body_%  : (Close-Open)/Open*100 -> how strong/decisive the candle was
      - Price_Gap      : Open - previous Close -> overnight gap
      - Volatility_10  : rolling std-dev of daily returns -> choppiness
      - Daily_Return_% : simple day-over-day % change
    """
    df["Candle_Body_%"] = (df["Close"] - df["Open"]) / df["Open"] * 100
    df["Price_Gap"] = df["Open"] - df["Close"].shift(1)
    df["Daily_Return_%"] = df["Close"].pct_change() * 100
    df["Volatility_10"] = df["Daily_Return_%"].rolling(window=10).std()
    return df


def add_derived_signals(df):
    """
    A couple of derived numbers that summarise the indicators above into
    features a model can use directly:
      - EMA_Cross        : EMA_9 - EMA_21 (positive = short-term bullish momentum)
      - Trend_vs_SMA200_% : % distance of Close from the 200-day trend line
                            (positive = price above long-term trend / uptrend regime)
      - VWAP_Deviation_%  : % distance of Close from rolling VWAP
                            (positive = trading above "fair value")
    """
    df["EMA_Cross"] = df["EMA_9"] - df["EMA_21"]
    df["Trend_vs_SMA200_%"] = (df["Close"] - df["SMA_200"]) / df["SMA_200"] * 100
    df["VWAP_Deviation_%"] = (df["Close"] - df["VWAP_20"]) / df["VWAP_20"] * 100
    return df


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the full focused indicator set:
    EMA 9, EMA 21, SMA 200, RSI 14, rolling VWAP(20), price-action features,
    and derived signals (EMA crossover, trend regime, VWAP deviation).
    """
    df = df.copy()
    df = add_ema(df, 9)
    df = add_ema(df, 21)
    df = add_sma(df, 200)
    df = add_rsi(df, 14)
    df = add_vwap(df, 20)
    df = add_price_action_features(df)
    df = add_derived_signals(df)
    return df


def add_targets(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Adds the prediction targets:
      - 'Target_Class'  : 1 if Close price 'horizon' days ahead is higher, else 0
      - 'Target_Return' : % return over 'horizon' days ahead

    NOTE: these look INTO THE FUTURE relative to each row, so the last
    `horizon` rows will have NaN targets — drop them before training.
    """
    df = df.copy()
    future_close = df["Close"].shift(-horizon)
    df["Target_Return"] = (future_close - df["Close"]) / df["Close"] * 100
    df["Target_Class"] = (future_close > df["Close"]).astype(int)
    return df


def prepare_feature_dataset(raw_df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Full feature pipeline: raw OHLCV -> indicators -> targets -> drop NaNs.
    NOTE: pass in data that ALREADY includes the SMA-200 warm-up buffer
    (see data_loader.fetch_stock_data / trim_to_period) or the SMA_200
    column will be NaN for the first ~200 rows and get dropped here.
    """
    df = build_all_features(df=raw_df)
    df = add_targets(df, horizon=horizon)
    df = df.dropna()
    return df


# The compact, trader-style feature set fed to the ML models.
FEATURE_COLUMNS = [
    "EMA_9", "EMA_21", "SMA_200", "RSI_14", "VWAP_20",
    "EMA_Cross", "Trend_vs_SMA200_%", "VWAP_Deviation_%",
    "Candle_Body_%", "Price_Gap", "Volatility_10",
]