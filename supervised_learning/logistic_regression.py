#!/usr/bin/env python
# coding: utf-8
import warnings
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')
warnings.filterwarnings('ignore')
# Load data
path = 'data/'
df = pd.read_csv(
    path + '/Bitcoin_08_08_2025-09_10_2025_historical_data_coinmarketcap.csv',
    sep=';',
    index_col=0
)
# Compute daily returns and lagged features. Target: 1 if return > 0 (up day), else 0.
df['return'] = df['close'].pct_change()
df['1_day_lag_returns'] = df['return'].shift(1)
df['2_day_lag_returns'] = df['return'].shift(2)
df['target'] = (df['return'] > 0).astype(int)
df = df.dropna()
# Split data chronologically (last 20% for testing).
features = ['1_day_lag_returns', '2_day_lag_returns']
X = df[features]
y = df['target']
test_frac = 0.20
split = int(len(df) * (1 - test_frac))
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]
print("Shapes ->",
      "X_train:", X_train.shape, "y_train:", y_train.shape,
      "X_test:", X_test.shape, "y_test:", y_test.shape)
# Logistic Regression Model
# LogisticRegression(C=1e5): High C prioritizes fitting training data over regularization.
logreg = LogisticRegression(C=1e5)
logreg.fit(X_train, y_train)
# Predict & Evaluate
# Predict up/down on test set. Evaluate with accuracy and confusion matrix.
y_pred = logreg.predict(X_test)
y_proba = logreg.predict_proba(X_test)[:, 1]  # P(up)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
print("Confusion matrix (rows=true, cols=pred):\n", cm)
# Decision Regions Plot
# - Axes: x = 1_day_lag_returns, y = 2_day_lag_returns
# - Shaded regions: Predicted class 0 (down) vs 1 (up); line is decision boundary (P(up)=0.5)
# - Dots: Test set points, colored by true class
# - Interpretation: Points in matching region = correct; opposite = error. Distance from boundary indicates confidence.
# - Slope: Trade-off between features (e.g., strong recent lag offsets weak prior lag).
f1, f2 = features
x1_lo, x1_hi = X[f1].quantile([0.01, 0.99])
x2_lo, x2_hi = X[f2].quantile([0.01, 0.99])
xx, yy = np.meshgrid(np.linspace(x1_lo, x1_hi, 200),
                     np.linspace(x2_lo, x2_hi, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = logreg.predict(grid).reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.25)
plt.scatter(X_test[f1], X_test[f2], c=y_test, s=25, edgecolor='k', alpha=0.85)
plt.xlabel(f1)
plt.ylabel(f2)
plt.title('Logistic Regression — Decision Regions (Test set)')
plt.savefig('outputs/logistic_regression', dpi=300, bbox_inches='tight')
print("Plot saved as 'decision_regions.png'")
plt.close()  # Close the figure to free memory
# Key takeaways
# What we built: A simple logistic regression that predicts next-day direction (0 = down, 1 = up) using lagged returns.
# No look-ahead: We used prior-day lags as features and kept the time order; the last 20% of data was our test set.
# Outputs: predict() gives class labels; predict_proba()[:, 1] gives P(up) for each day.
# Evaluation: Accuracy and the confusion matrix show overall hits and where the model confuses ups vs downs.
# Decision boundary: With two features, the model draws a straight line separating predicted up vs down regions.
# Limitations: Accuracy can be inflated by class imbalance; treat results as a teaching demo, not trading advice.
