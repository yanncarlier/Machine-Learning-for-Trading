#!/usr/bin/env python
# coding: utf-8
# Random Forest
#
# Random Forest, also called Random Decision Forest, is a method in machine learning capable of performing both regression and classification tasks. It is a type of ensemble learning that uses multiple learning algorithms for prediction.
#
# Random Forest comprises of decision trees, which are graphs of decisions representing their course of action or statistical probability. These multiple trees are plotted to a single tree called the Classification and Regression (CART) Model. To classify an object based on its attributes, each tree gives a classification that is said to vote for that class. The forest then chooses the classification with the maximum number of votes. For regression, it considers the average of the outputs for different trees.
#
# Working:
# 1. It assumes the number of cases as N. Then, randomly but with replacement, the sample of these N cases is taken out, which will be the training set.
# 2. Considering M to be the input variables, a number m is selected such that m < M. The best split between m and M is used to split the node. The value of m is held constant as the trees are grown.
# 3. Each tree is grown as large as possible.
# 4. By aggregating the predictions of n trees (i.e., majority votes for classification, the average for regression), random forest predicts the new data.
#
# Random Forest has certain advantages and disadvantages.
#
# Advantages:
# 1. This method balances the errors which are present in the dataset.
# 2. It is an effective method because it maintains accuracy even if it has to estimate the missing data.
# 3. Using the out-of-bag error estimate removes the need for a set-aside test set.
# 4. Random Forest helps in unsupervised clustering, data views, and outlier detection.
# 5. It reduces data management time and pre-processing tasks.
#
# Disadvantages:
# Disadvantages of the random forest may include its inability to be at par excellence for the regression problem as it does not give precise continuous nature predictions. It cannot predict beyond the range in the training data. Further, it does not provide complete control to the modeller.
#
# Applications of Random Forest:
# 1. It has many application in computational biology. Doctors can estimate the drug response to a particlar disease using this model.
# 2. This can be used to calculate a person's credit rating by comparing with other persons having similar traits.
#
# You can learn more about Random Forest and their application in trading in this article: https://blog.quantinsti.com/random-forest-algorithm-in-python/
# Import library
# For data manipulation
import numpy as np
import pandas as pd
# Import RandomForestClassifier and accuracy_score functions from sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
# Import Data
# We will read the daily data of Bitcoin, to create features.
# The data is stored in the directory 'data'
path = './data/'
# Read stock data from csv file
df = pd.read_csv(
    path + 'Bitcoin_08_08_2025-09_10_2025_historical_data_coinmarketcap.csv',
    sep=';',
    index_col=0
)
df.index = pd.to_datetime(df.index)
print(df.head())
print(df.tail())
# Independent Variables
# We will create independent variables which consist of 4 features. The features are:
# 1. Ratio of open and close price
# 2. Ratio of high and low price
# 3. 1-day lag returns
# 4. 2-day lag returns
# Create input features
df['return'] = df['close'].pct_change()
df['Open/Close'] = (df['open'] / df['close'])
df['High/Low'] = (df['high'] / df['low'])
df['1_day_lag_returns'] = df['return'].shift(1)
df['2_day_lag_returns'] = df['return'].shift(2)
# Drop NaN values
df.dropna(inplace=True)
# Store the features in a variable X
X = df[['Open/Close', 'High/Low', '1_day_lag_returns', '2_day_lag_returns']]
print(X.head())
# Dependent Variable
# When the next day's close price is greater than today's close price, we use 1 as a signal and else use -1. We will store this in the variable y, which is the dependent/target variable.
y = np.where(df['close'].shift(-1) > df['close'], 1, -1)
print(y[:10])
# Split the Dataset
# We will split the dataset into train and test samples. The train data consists of 75% of the total datasets. On the remaining, we will test the accuracy of the model.
# Training dataset length
split = int(len(df) * 0.75)
# Splitting the X and y into train and test datasets
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
# Train the Model
# We will use the RandomForestClassifier function from sklearn to train and fit the model.
# Create and fit the model on train dataset
clf = RandomForestClassifier(random_state=5)
model = clf.fit(X_train, y_train)
# Accuracy Score
# The model is trained on the training dataset. Now it's time to test the accuracy of the model on the test dataset. We will use accuracy_score function to test the accuracy.
print('Prediction Accuracy (%): ', round(accuracy_score(
    y_test, model.predict(X_test), normalize=True) * 100.0, 2))
# This is a very simple model with an accuracy of around 57.58% on the test dataset.
# Conclusion
# In this notebook, we learnt the functioning of the Random Forest Algorithm with the help of an example, along with the Python code to implement this strategy.
