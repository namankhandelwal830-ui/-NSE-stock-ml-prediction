"""
backtest.py
-----------
Simulates a simple long-only strategy driven by model predictions and
computes standard performance metrics (returns, Sharpe ratio, max drawdown,
win rate) compared against a buy-and-hold benchmark.

Strategy logic (simplified, for educational purposes):
  - If model predicts UP (1) for tomorrow -> hold a long position today
  - If model predicts DOWN (0) -> stay in cash (no short-selling)
"""

import numpy as np
import pandas as pd


def run_backtest(test_df: pd.DataFrame, predictions: np.ndarray,
                  probabilities: np.ndarray = None, confidence_threshold: float = 0.0,
                  transaction_cost_pct: float = 0.05,
                  initial_capital: float = 100000.0) -> pd.DataFrame:
    """
    Parameters
    ----------
    test_df : DataFrame slice (must be aligned/same length as predictions,
              must contain 'Close' column and be in chronological order)
    predictions : array of 0/1 (from a classifier) — 1 means "go long today"
    probabilities : optional array of the model's confidence (P of the
              predicted class) for each prediction. If provided together
              with confidence_threshold > 0, weak/unsure signals are
              filtered out (treated as "stay in cash") — this is closer
              to how a real trader would only act on high-conviction signals.
    confidence_threshold : minimum model confidence (0-1) required to act
              on a signal. 0 = act on every prediction (old behaviour).
    transaction_cost_pct : brokerage/slippage cost (%) charged EVERY time
              the strategy changes position (enters or exits) — ignoring
              this makes backtests look unrealistically profitable.
    initial_capital : starting portfolio value in INR

    Returns
    -------
    DataFrame with columns: Close, Daily_Return, Strategy_Return,
    Portfolio_Value, BuyHold_Value
    """
    df = test_df.copy().reset_index()
    df["Prediction"] = predictions

    # Optionally require the model to be confident before acting on a signal
    if probabilities is not None and confidence_threshold > 0:
        df["Confidence"] = probabilities
        confident_signal = (df["Prediction"] == 1) & (df["Confidence"] >= confidence_threshold)
        df["Prediction"] = confident_signal.astype(int)

    # Daily market return (buy & hold)
    df["Daily_Return"] = df["Close"].pct_change().fillna(0)

    # Strategy only earns the day's return when it predicted UP the day before,
    # so we use yesterday's prediction to decide whether we're "in the market" today.
    df["Position"] = df["Prediction"].shift(1).fillna(0)  # 1 = long, 0 = cash
    df["Strategy_Return"] = df["Daily_Return"] * df["Position"]

    # Transaction cost: charged whenever position CHANGES (entry or exit),
    # since every real trade has brokerage + slippage.
    position_changed = df["Position"].diff().abs().fillna(0) > 0
    cost = transaction_cost_pct / 100
    df["Strategy_Return"] = df["Strategy_Return"] - (position_changed.astype(int) * cost)

    # Cumulative portfolio value
    df["Portfolio_Value"] = initial_capital * (1 + df["Strategy_Return"]).cumprod()
    df["BuyHold_Value"] = initial_capital * (1 + df["Daily_Return"]).cumprod()

    return df


def compute_performance_metrics(backtest_df: pd.DataFrame, risk_free_rate: float = 0.06) -> dict:
    """
    Computes standard trading performance metrics.
    risk_free_rate: annual, e.g. 0.06 = 6% (approx Indian T-bill rate) for Sharpe ratio.
    """
    strategy_returns = backtest_df["Strategy_Return"]
    buyhold_returns = backtest_df["Daily_Return"]

    total_strategy_return = (backtest_df["Portfolio_Value"].iloc[-1] /
                              backtest_df["Portfolio_Value"].iloc[0] - 1) * 100
    total_buyhold_return = (backtest_df["BuyHold_Value"].iloc[-1] /
                             backtest_df["BuyHold_Value"].iloc[0] - 1) * 100

    # Annualised Sharpe ratio (assuming ~252 trading days/year)
    daily_rf = risk_free_rate / 252
    excess_returns = strategy_returns - daily_rf
    returns_std = strategy_returns.std()
    # If the strategy never took a position (or returns are constant), std ~ 0
    # and dividing by it blows up to a meaningless huge number. Report 0 instead.
    if returns_std < 1e-8:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = (excess_returns.mean() / returns_std) * np.sqrt(252)

    # Max drawdown
    cumulative = backtest_df["Portfolio_Value"]
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100

    # Win rate: % of days with a positive strategy return, among days we were in the market
    active_days = backtest_df[backtest_df["Position"] == 1]
    win_rate = (active_days["Strategy_Return"] > 0).mean() * 100 if len(active_days) > 0 else 0.0

    # Number of trades (position changes) — relevant now that each one has a cost
    num_trades = int((backtest_df["Position"].diff().abs().fillna(0) > 0).sum())

    return {
        "total_strategy_return_%": round(total_strategy_return, 2),
        "total_buyhold_return_%": round(total_buyhold_return, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "max_drawdown_%": round(max_drawdown, 2),
        "win_rate_%": round(win_rate, 2),
        "days_in_market": int(active_days.shape[0]),
        "total_days": int(backtest_df.shape[0]),
        "num_trades": num_trades,
    }
