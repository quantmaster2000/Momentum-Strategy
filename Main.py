from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt


def get_stock_data(ticker, start_date, end_date):

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    stock_data = yf.download(
        ticker,
        start=start_dt,
        end=end_dt,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    stock_data.reset_index(inplace=True)

    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.get_level_values(0)

    return stock_data

# Download data

TICKER = "QQQ"

df = get_stock_data(TICKER, "2023-07-14", "2026-07-14")

print(df.head())
print(df.columns)

# Feature Engineering

df["close_log_return"] = np.log(
    df["Close"] / df["Close"].shift(1)
)

df["close_log_return_lag_1"] = df["close_log_return"].shift(1)

df["close_log_return_dir_lag_1"] = np.sign(
    df["close_log_return_lag_1"]
)

df = df.dropna().reset_index(drop=True)

# Analysis

print("\nCorrelation\n")
print(
    df[
        [
            "close_log_return",
            "close_log_return_lag_1",
        ]
    ].corr()
)

print("\nGrouped statistics\n")
print(
    df.groupby("close_log_return_dir_lag_1")[
        "close_log_return"
    ].agg(["sum", "mean", "count"])
)

# Train/Test Split

split = int(len(df) * 0.75)

in_sample = df.iloc[:split]
out_sample = df.iloc[split:]

print("\nIn Sample\n")
print(
    in_sample.groupby("close_log_return_dir_lag_1")[
        "close_log_return"
    ].agg(["sum", "mean", "count"])
)

print("\nOut Sample\n")
print(
    out_sample.groupby("close_log_return_dir_lag_1")[
        "close_log_return"
    ].agg(["sum", "mean", "count"])
)

# Strategy

df["signal"] = df["close_log_return_dir_lag_1"]

df["trade_log_return"] = (
    df["signal"] * df["close_log_return"]
)

df["cum_trade_log_return"] = (
    df["trade_log_return"].cumsum()
)

# Performance

win_rate = (df["trade_log_return"] > 0).mean()

mean_return = df["trade_log_return"].mean()

std_return = df["trade_log_return"].std()

if std_return != 0:
    sharpe = (
        mean_return
        / std_return
        * np.sqrt(252)
    )
else:
    sharpe = np.nan

print(f"\nWin Rate : {win_rate:.2%}")
print(f"Mean Return : {mean_return:.6f}")
print(f"Std Return : {std_return:.6f}")
print(f"Sharpe : {sharpe:.2f}")

# Backtest

CAPITAL = 1000

df["post_trade_notional_value"] = (
    CAPITAL * np.exp(df["cum_trade_log_return"])
)

df["pre_trade_notional_value"] = (
    df["post_trade_notional_value"]
    .shift(1)
    .fillna(CAPITAL)
)

# Fees

TAKER_FEE_BPS = 4.1
MAKER_FEE_BPS = 1.2

TAKER_FEE_PCT = TAKER_FEE_BPS / 10000
MAKER_FEE_PCT = MAKER_FEE_BPS / 10000

df["entry_fee"] = (
    df["pre_trade_notional_value"]
    * TAKER_FEE_PCT
)

df["exit_fee"] = (
    df["post_trade_notional_value"]
    * TAKER_FEE_PCT
)

df["tx_fees"] = (
    df["entry_fee"]
    + df["exit_fee"]
)

df["cum_tx_fees"] = (
    df["tx_fees"].cumsum()
)

df["net_equity"] = (
    df["post_trade_notional_value"]
    - df["cum_tx_fees"]
)

# Plots

plt.figure(figsize=(12, 6))
plt.plot(df["net_equity"])
plt.title("Net Equity")
plt.grid(True)
plt.tight_layout()
plt.savefig("net_equity.png")

plt.figure(figsize=(12, 6))
plt.plot(df["cum_trade_log_return"])
plt.title("Cumulative Log Return")
plt.grid(True)
plt.tight_layout()
plt.savefig("cum_log_return.png")

plt.show()
