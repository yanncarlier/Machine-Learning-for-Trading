#!/usr/bin/env python
# coding: utf-8
"""
Support Vector Classifier Strategy on Bitcoin (adapted from S&P 500)
This script implements a Support Vector Classifier (SVC) to predict Bitcoin trends
and backtests a simple long/short strategy based on predictions.
Requirements:
- Python 3.x
- Libraries: scikit-learn, pandas, numpy, matplotlib
- Data: Bitcoin CSV in the '../data' folder (OHLC data from CoinMarketCap)
Steps:
1. Import libraries
2. Read Bitcoin data
3. Define explanatory variables (features)
4. Define target variable (signals: 1 for up, -1 for down)
5. Split data into train and test
6. Train SVC model
7. Predict signals and evaluate accuracy
8. Implement and plot strategy cumulative returns
Note: This uses a long/short strategy (signal 1/-1). For long-only, adjust target to 1/0.
"""
# Import libraries
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
# Suppress warnings
warnings.filterwarnings("ignore")
# Set plot style (fallback if seaborn not available)
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('default')

# Read Bitcoin Data
# CoinMarketCap CSV: semicolon-separated, WITH header (e.g., timeOpen,open,high,...), 6+ columns for OHLCV
csv_path = './data/Bitcoin_08_08_2025-09_10_2025_historical_data_coinmarketcap.csv'
df = pd.read_csv(
    csv_path, sep=';', header=0,  # FIXED: Use header=0 to skip header row
    index_col=0,  # FIXED: First column (timeOpen) as index
    parse_dates=True  # FIXED: Auto-parse index as datetime
)
# FIXED: Rename lowercase columns to match code (open -> Open, etc.)
df.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
# Keep only OHLCV for simplicity (assumes these are the first 5 columns after index)
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
print("Data loaded. Shape:", df.shape)
print(df.head())

# Define explanatory variables (features)
# Using simple ratios; add more indicators for better performance
df['Open/Close'] = df['Open'] / df['Close']
df['High/Low'] = df['High'] / df['Low']
X = df[['Open/Close', 'High/Low']]
print("\nFeatures created:")
print(X.head())

# Define target variable
# 1 if next Close > Close (long), -1 otherwise (short)
y_full = np.where(df['Close'].shift(-1) > df['Close'], 1, -1)
# Drop last NaN from shift(-1)
y = y_full[:-1]
X = X[:-1]  # Align X with y
print("\nTarget signals sample:")
print(y[:10])

# Split data into train and test (80/20)
split_percentage = 0.8
split = int(split_percentage * len(df)) - 1  # Adjust for dropped row
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]
print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# Train Support Vector Classifier
cls = SVC().fit(X_train, y_train)
print("\nSVC model trained.")

# Predict on test set
y_predict = cls.predict(X_test)
# Classifier accuracy
accuracy_test = accuracy_score(y_test, y_predict)
print(f"\nTest Accuracy: {accuracy_test:.2%}")

# Strategy implementation on test period
# Predict signals on full X (aligned)
df['Predicted_Signal'] = np.nan
df.iloc[:-1, df.columns.get_loc('Predicted_Signal')] = cls.predict(X)
df_test = df.iloc[split:].copy()  # Test period only (aligned)

# Calculate returns
df_test['Returns'] = df_test['Close'].pct_change()
df_test['Returns'] = df_test['Returns'].fillna(0)  # Fill first NaN

# Strategy returns (long/short: multiply by lagged signal)
df_test['Strategy_Returns'] = df_test['Returns'] * \
    df_test['Predicted_Signal'].shift(1)
df_test['Strategy_Returns'] = df_test['Strategy_Returns'].fillna(
    0)  # Fill initial NaN

# Cumulative returns (geometric)
df_test['Cumulative_Returns'] = (1 + df_test['Strategy_Returns']).cumprod()
df_test['Cumulative_Returns'] = df_test['Cumulative_Returns'].fillna(1)

# Plot cumulative returns
plt.figure(figsize=(15, 7))
plt.plot(df_test.index, df_test['Cumulative_Returns'],
         color='g', linewidth=2, label='SVC Strategy')
plt.plot(df_test.index, (1 + df_test['Returns']).cumprod().fillna(
    1), color='r', linewidth=1, label='Buy & Hold', alpha=0.7)
plt.title("Cumulative Returns: SVC Strategy on Bitcoin (Test Period)", fontsize=16)
plt.ylabel("Cumulative Returns")
plt.xlabel("Date")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Final cumulative return
final_return = (df_test['Cumulative_Returns'].iloc[-1] - 1) * 100
buy_hold_return = (
    (1 + df_test['Returns']).cumprod().fillna(1).iloc[-1] - 1) * 100
print(f"\nStrategy Cumulative Return (Test Period): {final_return:.2f}%")
print(f"Buy & Hold Cumulative Return (Test Period): {buy_hold_return:.2f}%")
print("\nCumulative Returns DataFrame (last 5 rows):")
print(df_test[['Cumulative_Returns']].tail())