#!/usr/bin/env python
# coding: utf-8
# K Nearest Neighbours (KNN)
#
# We will build a simple KNN classifier on trading data (Bitcoin). The goal is to predict next-day direction using lagged returns.
# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
# Load data
path = 'data/'
df = pd.read_csv(
    path + 'Bitcoin_08_08_2025-09_10_2025_historical_data_coinmarketcap.csv',
    sep=';',
    index_col=0
)
# Feature engineering
df['return'] = df['close'].pct_change()
df['1_day_lag_returns'] = df['return'].shift(1)
df['2_day_lag_returns'] = df['return'].shift(2)
df['target'] = (df['return'] > 0).astype(int)
df = df.dropna()
# Define features and target, split chronologically (last 20% for testing)
features = ['1_day_lag_returns', '2_day_lag_returns']
X = df[features]
y = df['target']
test_frac = 0.20
split = int(len(df) * (1 - test_frac))
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]
print('X_train:', X_train.shape, 'X_test:', X_test.shape)
# KNN Classification Model
# Create and fit the model
knn = KNeighborsClassifier()  # default k=5
knn.fit(X_train, y_train)
# Predict & Evaluate
# Predict on the test set, report test accuracy and the confusion matrix.
y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
print('Test accuracy:', round(acc, 4))
print('Confusion matrix (rows=true, cols=pred):\n', cm)
# Decision Regions
# A simple 2D plot showing predicted regions for class 0 and 1 over the lagged-return space with test points overlaid.
# Vibrant but readable (binary) palettes
# soft orange / soft blue (background)
cmap_light = ListedColormap(['#FFE4C7', '#CFE8FF'])
# vibrant orange / deep blue (points)
cmap_bold = ListedColormap(['#E86A0C', '#1E4FD7'])
# Use the test window’s full range (plus a tiny padding)
f1, f2 = features[0], features[1]
x1_lo, x1_hi = X_test[f1].min(), X_test[f1].max()
x2_lo, x2_hi = X_test[f2].min(), X_test[f2].max()
pad1 = 0.02 * (x1_hi - x1_lo) or 1e-6
pad2 = 0.02 * (x2_hi - x2_lo) or 1e-6
x1_lo, x1_hi = x1_lo - pad1, x1_hi + pad1
x2_lo, x2_hi = x2_lo - pad2, x2_hi + pad2
# Grid and predictions
xx, yy = np.meshgrid(
    np.linspace(x1_lo, x1_hi, 250),
    np.linspace(x2_lo, x2_hi, 250)
)
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
# Plot
plt.figure(figsize=(12, 9))
# Background regions for classes {0,1}
plt.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], cmap=cmap_bold, alpha=0.35)
# Test points colored by true class {0,1}
plt.scatter(
    X_test[f1], X_test[f2],
    c=y_test.to_numpy(), cmap=cmap_bold, vmin=0, vmax=1,
    s=32, edgecolor='k', linewidth=0.6, alpha=0.9, zorder=2
)
# Legend (true classes)
legend_handles = [
    Line2D([0], [0], marker='o', color='w', label='Class 0 (down)',
           markerfacecolor=cmap_bold(0), markeredgecolor='k', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Class 1 (up)',
           markerfacecolor=cmap_bold(1), markeredgecolor='k', markersize=8),
]
plt.legend(handles=legend_handles, loc='upper right', frameon=True)
plt.xlabel(f1)
plt.ylabel(f2)
plt.title('KNN — Decision Regions (Test set)')
plt.xlim(x1_lo, x1_hi)
plt.ylim(x2_lo, x2_hi)
plt.tight_layout()
plt.savefig('outputs/knn_classification.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'knn_decision_regions.png'")
plt.close()
# Interpreting the decision-region plot
# Axes: 1_day_lag_returns (x) vs 2_day_lag_returns (y).
# Background colors: Model’s predicted class in each area (0 = down, 1 = up).
# The boundary between colors is where the model is undecided.
# Dots: Test-set days, colored by the true class.
# A dot inside its own class region = correct; in the other region = error.
# Confidence intuition: Points farther from the boundary are typically more confident; near the line are uncertain.
# Why the patchy regions? KNN bases decisions on nearby points, so shapes can look “island-like” rather than a straight line.
