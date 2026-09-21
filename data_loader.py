"""
data_loader.py
---------------
Handles fetching live/historical NSE stock data using yfinance.

NSE tickers on Yahoo Finance need a ".NS" suffix, e.g. "RELIANCE.NS", "TCS.NS".
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def format_nse_ticker(symbol: str) -> str:
    """
    Ensures a raw NSE symbol (e.g. 'RELIANCE') is converted to the
    Yahoo Finance format (e.g. 'RELIANCE.NS').
    """
    symbol = symbol.strip().upper()
    if not symbol.endswith(".NS"):
        symbol += ".NS"
    return symbol


# Approximate calendar-day span for each display period the dashboard offers.
PERIOD_DAYS = {
    "6mo": 182,
    "1y": 365,
    "2y": 730,
    "3y": 1095,
    "5y": 1825,
}

# Extra calendar days fetched BEFORE the display window so that long-lookback
# indicators (like a 200-day SMA) already have enough warm-up history on day 1
# of the displayed period, instead of showing as NaN/blank for months.
WARMUP_BUFFER_DAYS = 320


def fetch_stock_data(symbol: str, period: str = "2y", interval: str = "1d",
                      buffer_days: int = WARMUP_BUFFER_DAYS) -> pd.DataFrame:
    """
    Fetches historical OHLCV data for a given NSE stock, INCLUDING an extra
    warm-up buffer before the requested period so indicators like SMA-200
    are valid from the very first displayed day (see trim_to_period()).

    Parameters
    ----------
    symbol : str
        NSE stock symbol, e.g. 'RELIANCE' or 'RELIANCE.NS'
    period : str
        Display period: '6mo', '1y', '2y', '3y', '5y'
    interval : str
        Candle interval. e.g. '1d', '1h', '15m'

    Returns
    -------
    pd.DataFrame with columns: Open, High, Low, Close, Volume (indexed by Date).
    NOTE: this includes the extra buffer window — call trim_to_period() before
    displaying/training on it.
    """
    ticker = format_nse_ticker(symbol)
    days = PERIOD_DAYS.get(period, 730)

    try:
        stock = yf.Ticker(ticker)
        end = datetime.now()
        start = end - timedelta(days=days + buffer_days)
        df = stock.history(start=start, end=end, interval=interval)

        if df.empty:
            raise ValueError(f"No data returned for {ticker}. Check the symbol or try a different period.")

        cols_to_keep = ["Open", "High", "Low", "Close", "Volume"]
        df = df[[c for c in cols_to_keep if c in df.columns]]
        df.index.name = "Date"
        return df

    except Exception as e:
        raise RuntimeError(f"Failed to fetch data for {ticker}: {e}")


def trim_to_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """
    Cuts a buffered DataFrame down to just the user-selected display period,
    dropping the extra warm-up rows fetched only so indicators could be
    computed correctly.
    """
    days = PERIOD_DAYS.get(period, 730)
    cutoff = df.index.max() - pd.Timedelta(days=days)
    return df[df.index >= cutoff]


def fetch_multiple_stocks(symbols: list, period: str = "2y", interval: str = "1d") -> dict:
    """
    Fetches data for multiple NSE stocks at once.
    Returns a dict: {symbol: DataFrame}  (each already trimmed to `period`)
    """
    data = {}
    for sym in symbols:
        try:
            raw = fetch_stock_data(sym, period=period, interval=interval)
            data[sym] = trim_to_period(raw, period)
        except RuntimeError as e:
            print(f"[WARN] Skipping {sym}: {e}")
    return data


def get_company_info(symbol: str) -> dict:
    """
    Fetches company fundamentals for display on the dashboard's
    Fundamentals tab and header.

    NOTE: Yahoo Finance sometimes rate-limits or blocks requests from cloud
    hosting IPs (e.g. Streamlit Community Cloud), even though price history
    (fetch_stock_data) keeps working fine — .info is a separate, stricter
    endpoint. When that happens we raise here so the caller can show a
    friendly fallback message instead of a grid full of "N/A".
    """
    ticker = format_nse_ticker(symbol)
    stock = yf.Ticker(ticker)
    info = stock.info

    if not info or not info.get("longName"):
        raise RuntimeError(
            f"Yahoo Finance didn't return company fundamentals for {ticker} "
            f"right now (this can happen from shared cloud IPs) — price data "
            f"and predictions are unaffected."
        )

    return {
        # --- Overview ---
        "name": info.get("longName", symbol),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "current_price": info.get("currentPrice", "N/A"),
        "market_cap": info.get("marketCap", "N/A"),
        "beta": info.get("beta", "N/A"),
        "week_52_high": info.get("fiftyTwoWeekHigh", "N/A"),
        "week_52_low": info.get("fiftyTwoWeekLow", "N/A"),
        "avg_volume": info.get("averageVolume", "N/A"),
        "shares_outstanding": info.get("sharesOutstanding", "N/A"),

        # --- Valuation ---
        "pe_ratio": info.get("trailingPE", "N/A"),
        "forward_pe": info.get("forwardPE", "N/A"),
        "price_to_book": info.get("priceToBook", "N/A"),
        "ev_to_ebitda": info.get("enterpriseToEbitda", "N/A"),
        "ev_to_revenue": info.get("enterpriseToRevenue", "N/A"),
        "peg_ratio": info.get("pegRatio", "N/A"),

        # --- Profitability ---
        "roe": info.get("returnOnEquity", "N/A"),
        "roa": info.get("returnOnAssets", "N/A"),
        "operating_margin": info.get("operatingMargins", "N/A"),
        "profit_margin": info.get("profitMargins", "N/A"),
        "gross_margin": info.get("grossMargins", "N/A"),

        # --- Financial health ---
        "debt_to_equity": info.get("debtToEquity", "N/A"),
        "current_ratio": info.get("currentRatio", "N/A"),
        "quick_ratio": info.get("quickRatio", "N/A"),
        "total_cash": info.get("totalCash", "N/A"),
        "total_debt": info.get("totalDebt", "N/A"),
        "free_cash_flow": info.get("freeCashflow", "N/A"),

        # --- Per-share ---
        "eps": info.get("trailingEps", "N/A"),
        "forward_eps": info.get("forwardEps", "N/A"),
        "book_value": info.get("bookValue", "N/A"),
        "revenue_per_share": info.get("revenuePerShare", "N/A"),
        "cash_per_share": info.get("totalCashPerShare", "N/A"),

        # --- Growth ---
        "revenue_growth": info.get("revenueGrowth", "N/A"),
        "earnings_growth": info.get("earningsGrowth", "N/A"),
        "earnings_qtr_growth": info.get("earningsQuarterlyGrowth", "N/A"),

        # --- Dividend & ownership ---
        "dividend_yield": info.get("dividendYield", "N/A"),
        "payout_ratio": info.get("payoutRatio", "N/A"),
        "pct_held_institutions": info.get("heldPercentInstitutions", "N/A"),
        "pct_held_insiders": info.get("heldPercentInsiders", "N/A"),
    }


if __name__ == "__main__":
    # quick manual test
    df = fetch_stock_data("RELIANCE", period="6mo")
    print(f"Fetched (with buffer): {len(df)} rows, from {df.index.min()} to {df.index.max()}")
    trimmed = trim_to_period(df, "6mo")
    print(f"Trimmed to display period: {len(trimmed)} rows, from {trimmed.index.min()} to {trimmed.index.max()}")
