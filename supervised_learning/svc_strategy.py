#!/usr/bin/env python
# coding: utf-8
# Support Vector Classifier Strategy Code
#
# In this notebook, you will learn to create a Support Vector Classifier (SVC) algorithm on Bitcoin data.
# Import the libraries
import warnings
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# For data manipulation
import pandas as pd
import numpy as np
# To plot
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')
# To ignore warnings
warnings.filterwarnings("ignore")
# Read S&P 500 Data
# The data is stored in the directory 'data'
path = './data/'
# Read the csv file using read_csv method of pandas
df = pd.read_csv(
    path + 'Bitcoin_08_08_2025-09_10_2025_historical_data_coinmarketcap.csv',
    sep=';',
    index_col=0
)
# Convert index to datetime format
df.index = pd.to_datetime(df.index)
df.head()
# Define the explanatory variables
# Create predictor variables
df['Open/Close'] = df.open / df.close
df['High/Low'] = df.high / df.low
# Store all predictor variables in a variable X
X = df[['Open/Close', 'High/Low']]
X.head()
# Define the target variable
# Target variables
y = np.where(df['close'].shift(-1) > df['close'], 1, -1)
# Print y
print(y)
# Split the data into train and test
# Define the split percentage
split_percentage = 0.8
split = int(split_percentage * len(df))
# Train data set
X_train = X[:split]
y_train = y[:split]
# Test data set
X_test = X[split:]
y_test = y[split:]
# Support Vector Classifier (SVC)
# Support vector classifier
cls = SVC().fit(X_train, y_train)
# Predict Signals
# We will use the predict method on the cls variable to predict the signals.
y_predict = cls.predict(X_test)
# Classifier accuracy
# train and test accuracy
accuracy_test = accuracy_score(y_test, y_predict)
print('Accuracy:{: .2f}%'.format(accuracy_test * 100))
# An accuracy of 50%+ in test data suggests that the classifier model is effective.
# Strategy implementation
# Predicted Signal
df['Predicted_Signal'] = cls.predict(X)
df = df[split:]
# Calculate daily returns
df['Returns'] = df.close.pct_change()
# Calculate strategy returns
df['Strategy_Returns'] = df.Returns * df.Predicted_Signal.shift(1)
# Calculate geometric returns
df['cumulative_returns'] = (
    df['Strategy_Returns'] + 1).cumprod().shift().fillna(1)
# Set the title and axis labels
plt.title("Cumulative Returns Plot", fontsize=16)
plt.ylabel("Cumulative Returns")
plt.xlabel("Date")
# Plot geometric returns
df['cumulative_returns'].plot(figsize=(15, 7), color='g')
plt.savefig('./outputs/svc_strategy.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'svc_strategy.png'")
plt.show()
# As seen from the graph, the strategy generates a return of approximately 10.38% in the test data set.
#
# This notebook shows how you can implement the SVC strategy step by step with Python. You can train and test your model and also check the strategy implementation with a visual representation (graph).
#
# ## Tweak the code
# You can tweak the code in the following ways:
#
# 1. Use different datasets: Backtest and try out the model on different datasets!
# 2. Features: Create your features using different indicators that could improve the prediction accuracy.
#
# In the next units, you will be able to practice some important concepts learned in this section. In the next section, you will learn Natural Language Processing (NLP) and its implementation in sentiment analysis.
