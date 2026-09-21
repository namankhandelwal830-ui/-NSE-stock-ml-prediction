"""
app.py
------
Streamlit dashboard: the entry point for the whole project.

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_loader import fetch_stock_data, trim_to_period, get_company_info
from preprocessing import clean_pipeline
from features import prepare_feature_dataset, build_all_features, FEATURE_COLUMNS
from models import train_all_models, cross_validate_models
from backtest import run_backtest, compute_performance_metrics
from support_resistance import detect_support_resistance
from stock_universe import STOCK_UNIVERSE, LAST_UPDATED, get_display_options, parse_symbol_from_display
from report_generator import generate_pdf_report


st.set_page_config(page_title="NSE Stock Analysis & ML Prediction", layout="wide")


def fmt_pct(value):
    if value in (None, "N/A") or not isinstance(value, (int, float)):
        return "N/A"
    return f"{value*100:.2f}%"


def fmt_num(value, prefix=""):
    if value in (None, "N/A") or not isinstance(value, (int, float)):
        return "N/A"
    if abs(value) >= 1e7:
        return f"{prefix}{value/1e7:.2f} Cr"
    return f"{prefix}{value:,.2f}"


# ---------------------------------------------------------------
# CACHED DATA/PIPELINE FUNCTIONS
# Caching avoids re-hitting Yahoo Finance and re-running the whole
# cleaning/feature/model pipeline every time the user just tweaks an
# unrelated slider — only re-runs when symbol/period/horizon/train_size
# actually change (or after 15 minutes, so prices don't go stale).
# ---------------------------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def cached_company_info(symbol: str):
    return get_company_info(symbol)


@st.cache_data(ttl=900, show_spinner=False)
def cached_pipeline(symbol: str, period: str, horizon: int):
    """Fetch -> clean -> engineer features, all cached together."""
    raw_df_buffered = fetch_stock_data(symbol, period=period)
    clean_df_buffered = clean_pipeline(raw_df_buffered)
    feature_df_buffered = prepare_feature_dataset(clean_df_buffered, horizon=horizon)
    clean_df = trim_to_period(clean_df_buffered, period)
    feature_df = trim_to_period(feature_df_buffered, period)
    full_indicators_df = build_all_features(clean_df_buffered).dropna()
    return clean_df, feature_df, full_indicators_df


# ---------------------------------------------------------------
# SIDEBAR — user inputs
# ---------------------------------------------------------------
st.sidebar.title("⚙️ Configuration")

stock_options = ["Type manually below..."] + get_display_options()
picked = st.sidebar.selectbox("Popular Stocks (NSE)", stock_options, index=1)
default_symbol = parse_symbol_from_display(picked) if picked != "Type manually below..." else "RELIANCE"
symbol = st.sidebar.text_input("Or type any NSE Symbol", value=default_symbol).strip().upper()

st.sidebar.caption(f"Popular list last reviewed: {LAST_UPDATED}. Any valid NSE symbol works, listed or not.")

period = st.sidebar.selectbox("Historical Period", ["6mo", "1y", "2y", "3y", "5y"], index=2)
horizon = st.sidebar.slider("Prediction Horizon (days ahead)", 1, 10, 1)
train_size = st.sidebar.slider("Train/Test Split (train %)", 0.6, 0.9, 0.8, step=0.05)
confidence_threshold = st.sidebar.slider(
    "Min. Model Confidence to Trade", 0.5, 0.9, 0.55, step=0.05,
    help="Backtest only 'buys' when the model is at least this confident — filters out weak/unsure signals."
)
transaction_cost = st.sidebar.slider(
    "Transaction Cost per Trade (%)", 0.0, 0.5, 0.05, step=0.05,
    help="Brokerage + slippage charged every time the strategy enters/exits a position."
)
run_button = st.sidebar.button("🚀 Run Analysis")

st.title("📈 NSE Stock Analysis & ML Prediction System")
st.caption("Live NSE data · EMA 9/21 · SMA 200 · RSI · VWAP · Support/Resistance · "
           "Random Forest / SVM / Linear Regression · Realistic Backtesting")


# ---------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------
if run_button:
    if not symbol:
        st.error("Please enter or select a stock symbol before running the analysis.")
        st.stop()

    with st.spinner(f"Fetching live data for {symbol}..."):
        try:
            clean_df, feature_df, full_indicators_df = cached_pipeline(symbol, period, horizon)
        except RuntimeError:
            st.error(
                f"⚠️ Couldn't fetch data for **{symbol}**. This usually means either:\n\n"
                f"- The symbol is misspelled or isn't listed on the NSE (try just the base "
                f"symbol, e.g. `RELIANCE` not `RELIANCE.NS` or `RELIANCE LTD`)\n"
                f"- Yahoo Finance is temporarily unavailable — try again in a minute\n\n"
                f"Tip: pick a stock from the **Popular Stocks** dropdown in the sidebar to "
                f"guarantee a valid symbol."
            )
            st.stop()
        except Exception as e:
            st.error(f"⚠️ Something unexpected went wrong while processing {symbol}: {e}")
            st.stop()

    # --- Company info header ---
    info = None
    try:
        info = cached_company_info(symbol)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Company", info["name"])
        c2.metric("Sector", info["sector"])
        c3.metric("Current Price (₹)", info["current_price"])
        c4.metric("P/E Ratio", info["pe_ratio"])
    except Exception:
        st.info("Company metadata unavailable for this symbol; continuing with price data only.")

    if len(feature_df) < 50:
        st.error(
            "Not enough data after cleaning/feature engineering to train reliable models. "
            "Try a longer historical period (e.g. switch from 6mo to 1y or 2y)."
        )
        st.stop()

    # ---------------- Train models (shared across all tabs) ----------------
    with st.spinner("Training Random Forest, SVM, and Linear Regression models..."):
        results = train_all_models(feature_df, FEATURE_COLUMNS, train_size=train_size)

    # ---------------- Cross-validation (more reliable accuracy estimate) ----------------
    with st.spinner("Running walk-forward cross-validation..."):
        cv_results = cross_validate_models(feature_df, FEATURE_COLUMNS, n_splits=5)

    # ---------------- Support / Resistance ----------------
    sr_levels = detect_support_resistance(clean_df, order=5, tolerance_pct=1.5, max_levels=4)

    # ---------------- LIVE PREDICTION ----------------
    st.markdown("---")
    st.subheader(f"🔮 Live Prediction — What do the models say about {symbol} next?")

    latest_row = full_indicators_df[FEATURE_COLUMNS].iloc[[-1]]
    latest_date = full_indicators_df.index[-1]

    rf_model = results["random_forest"]["model"]
    svm_model = results["svm"]["model"]
    svm_scaler = results["svm"]["scaler"]

    rf_pred = rf_model.predict(latest_row)[0]
    rf_proba = rf_model.predict_proba(latest_row)[0]

    svm_latest_scaled = svm_scaler.transform(latest_row)
    svm_pred = svm_model.predict(svm_latest_scaled)[0]
    svm_proba = svm_model.predict_proba(svm_latest_scaled)[0]

    lin_reg_model = results["linear_regression"]["model"]
    predicted_return = lin_reg_model.predict(latest_row)[0]

    st.caption(f"Based on indicators as of the last available trading day: **{latest_date.date()}**, "
               f"forecasting **{horizon} day(s)** ahead.")

    p1, p2, p3 = st.columns(3)
    with p1:
        direction = "📈 UP" if rf_pred == 1 else "📉 DOWN"
        confidence = rf_proba[1] if rf_pred == 1 else rf_proba[0]
        st.metric("Random Forest says", direction, f"{confidence*100:.1f}% confidence")
    with p2:
        direction = "📈 UP" if svm_pred == 1 else "📉 DOWN"
        confidence = svm_proba[1] if svm_pred == 1 else svm_proba[0]
        st.metric("SVM says", direction, f"{confidence*100:.1f}% confidence")
    with p3:
        st.metric("Linear Regression predicts return", f"{predicted_return:+.2f}%")

    if rf_pred == svm_pred:
        st.success(f"✅ Both models **agree**: {'UP 📈' if rf_pred == 1 else 'DOWN 📉'} — a somewhat stronger (though still not guaranteed) signal.")
    else:
        st.warning("⚠️ Models **disagree** with each other — a sign the signal is weak/uncertain right now.")

    st.warning(
        "This is a model guess based on historical patterns, not financial advice. "
        "Treat this as a pipeline demo, not a signal to actually trade on."
    )

    # ---------------- Backtest (computed once, used in Tab 4 and the PDF report) ----------------
    test_df = results["test_df"]
    rf_preds = results["random_forest"]["metrics"]["predictions"]
    rf_probas = results["random_forest"]["metrics"]["probabilities"]
    bt_df = run_backtest(test_df, rf_preds, probabilities=rf_probas,
                          confidence_threshold=confidence_threshold,
                          transaction_cost_pct=transaction_cost)
    backtest_perf = compute_performance_metrics(bt_df)

    # ---------------- PDF Report download ----------------
    try:
        pdf_buffer = generate_pdf_report(
            symbol=symbol, info=info or {}, latest_date=str(latest_date.date()), horizon=horizon,
            rf_pred=rf_pred, rf_confidence=(rf_proba[1] if rf_pred == 1 else rf_proba[0]),
            svm_pred=svm_pred, svm_confidence=(svm_proba[1] if svm_pred == 1 else svm_proba[0]),
            predicted_return=predicted_return,
            rf_metrics=results["random_forest"]["metrics"], svm_metrics=results["svm"]["metrics"],
            linreg_metrics=results["linear_regression"]["metrics"],
            backtest_perf=backtest_perf, cv_results=cv_results,
        )
        st.download_button(
            "📄 Download PDF Report", data=pdf_buffer,
            file_name=f"{symbol}_analysis_report.pdf", mime="application/pdf"
        )
    except Exception:
        st.caption("PDF report generation is temporarily unavailable for this run.")

    # =============================================================
    # TAB LAYOUT
    # =============================================================
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Price Chart", "🤖 Model Performance", "🔍 Feature Importance",
         "💰 Backtest Results", "🏦 Fundamentals"]
    )

    # ---------------- TAB 1: Price chart — EMA 9/21, SMA 200, VWAP, RSI, S/R, Volume ----------------
    with tab1:
        st.subheader(f"{symbol} — Price Action & Key Indicators")

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
            row_heights=[0.62, 0.13, 0.25],
        )

        fig.add_trace(go.Candlestick(
            x=clean_df.index, open=clean_df["Open"], high=clean_df["High"],
            low=clean_df["Low"], close=clean_df["Close"], name="Price",
            increasing_line_color="#26A69A", decreasing_line_color="#EF5350"
        ), row=1, col=1)

        fig.add_trace(go.Scatter(x=feature_df.index, y=feature_df["EMA_9"],
                                  line=dict(color="#00BFFF", width=1.2), name="EMA 9"), row=1, col=1)
        fig.add_trace(go.Scatter(x=feature_df.index, y=feature_df["EMA_21"],
                                  line=dict(color="#FFA500", width=1.2), name="EMA 21"), row=1, col=1)
        fig.add_trace(go.Scatter(x=feature_df.index, y=feature_df["SMA_200"],
                                  line=dict(color="#FF6B6B", width=1.6), name="SMA 200"), row=1, col=1)
        fig.add_trace(go.Scatter(x=feature_df.index, y=feature_df["VWAP_20"],
                                  line=dict(color="#B266FF", width=1.1, dash="dot"), name="VWAP (20)"), row=1, col=1)

        # Support & Resistance — only label MAJOR levels (Minor ones still draw,
        # just fainter and unlabeled, so the chart doesn't get cluttered with text)
        for r in sr_levels["resistance"]:
            is_major = r["strength"] == "Major"
            hline_kwargs = dict(
                y=r["price"], line_dash="dash",
                line_color="rgba(239,83,80,0.75)" if is_major else "rgba(239,83,80,0.25)",
                line_width=1.6 if is_major else 1,
            )
            if is_major:
                hline_kwargs.update(
                    annotation_text=f"R  ₹{r['price']:.0f}",
                    annotation_font=dict(size=10, color="rgba(239,83,80,0.9)"),
                    annotation_position="right",
                )
            fig.add_hline(row=1, col=1, **hline_kwargs)

        for s in sr_levels["support"]:
            is_major = s["strength"] == "Major"
            hline_kwargs = dict(
                y=s["price"], line_dash="dash",
                line_color="rgba(38,166,154,0.75)" if is_major else "rgba(38,166,154,0.25)",
                line_width=1.6 if is_major else 1,
            )
            if is_major:
                hline_kwargs.update(
                    annotation_text=f"S  ₹{s['price']:.0f}",
                    annotation_font=dict(size=10, color="rgba(38,166,154,0.9)"),
                    annotation_position="right",
                )
            fig.add_hline(row=1, col=1, **hline_kwargs)

        # Current price — thin white dotted line, small label, positioned to
        # avoid clashing with the y-axis tick labels
        last_close = clean_df["Close"].iloc[-1]
        fig.add_hline(
            y=last_close, line_dash="dot", line_color="rgba(255,255,255,0.6)", line_width=1,
            annotation_text=f"₹{last_close:.2f}", annotation_font=dict(size=10, color="white"),
            annotation_position="top left", row=1, col=1
        )

        # Volume bars, colored by up/down day
        vol_colors = ["#26A69A" if c >= o else "#EF5350" for o, c in zip(clean_df["Open"], clean_df["Close"])]
        fig.add_trace(go.Bar(x=clean_df.index, y=clean_df["Volume"], marker_color=vol_colors,
                              name="Volume", showlegend=False, opacity=0.8), row=2, col=1)

        fig.add_trace(go.Scatter(x=feature_df.index, y=feature_df["RSI_14"],
                                  line=dict(color="#00E5A0", width=1.3), name="RSI 14"), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(239,83,80,0.5)", line_width=1, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(38,166,154,0.5)", line_width=1, row=3, col=1)

        fig.update_yaxes(title_text="Price (₹)", title_font=dict(size=11), row=1, col=1)
        fig.update_yaxes(title_text="Volume", title_font=dict(size=11), row=2, col=1)
        fig.update_yaxes(title_text="RSI", title_font=dict(size=11), range=[0, 100], row=3, col=1)

        fig.update_xaxes(
            rangeslider=dict(visible=True, thickness=0.04),
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all", label="All"),
                ],
                bgcolor="rgba(38,39,48,0.9)", font=dict(size=11)
            ),
            row=3, col=1
        )

        fig.update_layout(
            height=820,
            hovermode="x unified",
            template="plotly_dark",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            margin=dict(t=45, b=10, l=10, r=70),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                bgcolor="rgba(0,0,0,0)", font=dict(size=11)
            ),
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
            yaxis2=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis3=dict(gridcolor="rgba(255,255,255,0.07)"),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "🔴 Resistance · 🟢 Support (darker/labeled = **Major**, more historically significant) · "
            "⚪ dotted = current price · Drag the range slider or use the 1M–All buttons to zoom."
        )

    # ---------------- TAB 2: Model performance ----------------
    with tab2:
        st.subheader("Classification Models — Predicting Next-Day Direction (Up/Down)")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🌳 Random Forest Classifier")
            m = results["random_forest"]["metrics"]
            st.metric("Accuracy", f"{m['accuracy']*100:.2f}%")
            st.metric("Precision", f"{m['precision']*100:.2f}%")
            st.metric("Recall", f"{m['recall']*100:.2f}%")
            st.metric("F1 Score", f"{m['f1_score']*100:.2f}%")
            st.write("Confusion Matrix (rows=actual, cols=predicted):")
            st.dataframe(pd.DataFrame(m["confusion_matrix"],
                                       index=["Actual Down", "Actual Up"],
                                       columns=["Pred Down", "Pred Up"]))

        with col2:
            st.markdown("### 🧮 SVM Classifier")
            m = results["svm"]["metrics"]
            st.metric("Accuracy", f"{m['accuracy']*100:.2f}%")
            st.metric("Precision", f"{m['precision']*100:.2f}%")
            st.metric("Recall", f"{m['recall']*100:.2f}%")
            st.metric("F1 Score", f"{m['f1_score']*100:.2f}%")
            st.write("Confusion Matrix (rows=actual, cols=predicted):")
            st.dataframe(pd.DataFrame(m["confusion_matrix"],
                                       index=["Actual Down", "Actual Up"],
                                       columns=["Pred Down", "Pred Up"]))

        st.subheader("Regression Model — Predicting % Return Magnitude")
        m = results["linear_regression"]["metrics"]
        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", f"{m['rmse']:.3f}")
        c2.metric("MAE", f"{m['mae']:.3f}")
        c3.metric("R² Score", f"{m['r2_score']:.3f}")

        st.markdown("---")
        st.subheader(f"🔁 Cross-Validation — {cv_results['n_splits']}-Fold Walk-Forward")
        st.caption(
            "A single train/test split can look good or bad by chance depending on which "
            "time window it happened to test on. This instead trains/tests across several "
            "different chronological windows and reports the average — a more reliable "
            "estimate of real-world performance, with the ± showing how much it varies."
        )
        cv1, cv2 = st.columns(2)
        with cv1:
            st.metric("Random Forest — Mean CV Accuracy",
                      f"{cv_results['rf_mean_accuracy']*100:.2f}%",
                      f"±{cv_results['rf_std']*100:.2f}% std dev")
        with cv2:
            st.metric("SVM — Mean CV Accuracy",
                      f"{cv_results['svm_mean_accuracy']*100:.2f}%",
                      f"±{cv_results['svm_std']*100:.2f}% std dev")

        st.caption(
            "Note: short-term equity direction is notoriously hard to predict — "
            "accuracy near 50-58% is realistic and expected, not a bug. Class weighting "
            "was applied so the models can't just always guess the majority class. "
            "Treat this as a learning/portfolio project, not a live trading signal."
        )

    # ---------------- TAB 3: Feature importance ----------------
    with tab3:
        st.subheader("Which indicators mattered most? (Random Forest)")
        importance_df = results["random_forest"]["importance"]
        fig_imp = go.Figure(go.Bar(
            x=importance_df["importance"], y=importance_df["feature"],
            orientation="h"
        ))
        fig_imp.update_layout(height=450, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_imp, use_container_width=True)

    # ---------------- TAB 4: Backtest ----------------
    with tab4:
        st.subheader("Strategy Backtest (Random Forest signal) vs Buy & Hold")
        st.caption(
            f"Only acts when model confidence ≥ {confidence_threshold*100:.0f}% · "
            f"{transaction_cost:.2f}% cost charged per trade — a more realistic simulation "
            "than a cost-free, always-act backtest."
        )

        perf = backtest_perf

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Strategy Return", f"{perf['total_strategy_return_%']}%")
        c2.metric("Buy & Hold Return", f"{perf['total_buyhold_return_%']}%")
        c3.metric("Sharpe Ratio", perf["sharpe_ratio"])
        c4.metric("Max Drawdown", f"{perf['max_drawdown_%']}%")
        c5.metric("Win Rate", f"{perf['win_rate_%']}%")
        c6.metric("Trades Taken", perf["num_trades"])

        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=bt_df["Date"], y=bt_df["Portfolio_Value"],
                                     name="ML Strategy", line=dict(color="#26A69A")))
        fig_bt.add_trace(go.Scatter(x=bt_df["Date"], y=bt_df["BuyHold_Value"],
                                     name="Buy & Hold", line=dict(color="gray", dash="dot")))
        fig_bt.update_layout(height=450, yaxis_title="Portfolio Value (₹)", hovermode="x unified")
        st.plotly_chart(fig_bt, use_container_width=True)

    # ---------------- TAB 5: Fundamentals ----------------
    with tab5:
        st.subheader(f"{symbol} — Fundamental Snapshot")
        if info is None:
            st.info("Fundamental data unavailable for this symbol.")
        else:
            f1, f2, f3, f4 = st.columns(4)
            f1.metric("Market Cap (₹)", fmt_num(info.get("market_cap")))
            f2.metric("P/E (Trailing)", info.get("pe_ratio", "N/A"))
            f3.metric("P/E (Forward)", info.get("forward_pe", "N/A"))
            f4.metric("EPS (₹)", info.get("eps", "N/A"))

            g1, g2, g3, g4 = st.columns(4)
            g1.metric("Book Value (₹)", info.get("book_value", "N/A"))
            g2.metric("Dividend Yield", fmt_pct(info.get("dividend_yield")))
            g3.metric("ROE", fmt_pct(info.get("roe")))
            g4.metric("Debt/Equity", info.get("debt_to_equity", "N/A"))

            h1, h2, h3, h4 = st.columns(4)
            h1.metric("Beta", info.get("beta", "N/A"))
            h2.metric("52W High (₹)", info.get("week_52_high", "N/A"))
            h3.metric("52W Low (₹)", info.get("week_52_low", "N/A"))
            h4.metric("Profit Margin", fmt_pct(info.get("profit_margin")))

            st.metric("Revenue Growth (YoY)", fmt_pct(info.get("revenue_growth")))

            st.caption(
                "Fundamentals are sourced live from Yahoo Finance at run time and reflect "
                "the company's overall financial health — separate from the technical/ML "
                "analysis above, which only looks at price and volume patterns."
            )

else:
    st.info("👈 Pick a stock from the sidebar (or type any NSE symbol) and click **Run Analysis**.")