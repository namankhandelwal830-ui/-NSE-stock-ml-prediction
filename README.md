# NSE Stock Analysis & ML Prediction System

An end-to-end ML pipeline that ingests live NSE (National Stock Exchange, India) data,
engineers technical indicators, trains classification/regression models, and serves
everything through an interactive Streamlit dashboard with backtesting.

## Stack
Python · Scikit-Learn · Pandas · Streamlit · yfinance · Plotly

## Project structure
```
nse_ml_project/
├── data_loader.py      # Live NSE data ingestion (yfinance)
├── preprocessing.py    # Missing value handling, outlier detection/treatment, normalisation
├── features.py         # 11 technical indicators + target generation
├── models.py             # Random Forest, SVM, Linear Regression training & evaluation
├── backtest.py           # Strategy backtesting engine & performance metrics
├── app.py                 # Streamlit dashboard (main entry point)
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard
streamlit run app.py
```

The app opens at `http://localhost:8501`. Enter any NSE symbol (e.g. `RELIANCE`,
`TCS`, `INFY`, `HDFCBANK`, `SBIN`) in the sidebar and click **Run Analysis**.

## How the pipeline works

1. **Data ingestion** (`data_loader.py`) — pulls historical OHLCV data for the chosen
   NSE stock via `yfinance` (auto-appends the `.NS` suffix Yahoo Finance requires).
2. **Preprocessing** (`preprocessing.py`) — forward-fills missing values, detects
   outliers with the IQR method, and winsorizes (caps) them.
3. **Feature engineering** (`features.py`) — computes 11 technical indicators
   (SMA, EMA, RSI, MACD + signal, Bollinger Bands, ATR, Stochastic %K, ROC, OBV,
   rolling volatility, overnight price gap) and generates two prediction targets:
   next-period direction (classification) and next-period % return (regression).
4. **Modeling** (`models.py`) — trains a Random Forest classifier, an SVM classifier
   (with feature scaling), and a Linear Regression model, using a **chronological**
   (not random) train/test split to avoid look-ahead bias.
5. **Backtesting** (`backtest.py`) — simulates a long/cash strategy driven by the
   Random Forest's predictions and compares it against buy-and-hold, reporting
   total return, Sharpe ratio, max drawdown, and win rate.
6. **Dashboard** (`app.py`) — ties it all together: candlestick chart with
   indicators, model performance metrics, feature importance, and backtest curve.

## Honest caveat (worth mentioning in interviews)

Short-term equity price direction is very hard to predict — expect classifier
accuracy in the 50-58% range on most NSE large caps, which is realistic, not a bug.
This project's value is in demonstrating the *pipeline* (data engineering,
preprocessing rigor, avoiding leakage, backtesting discipline) rather than claiming
a profitable trading edge. Framing it this way in interviews will land better than
overselling the predictive accuracy.

## Possible extensions
- Add LSTM/GRU sequence models for comparison against the classical baselines
- Walk-forward (rolling-window) validation instead of a single train/test split
- Add transaction costs & slippage to the backtest for realism
- Deploy on Streamlit Community Cloud for a live demo link
