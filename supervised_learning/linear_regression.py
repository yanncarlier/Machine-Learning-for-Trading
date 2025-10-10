#!/usr/bin/env python
# coding: utf-8
"""
Linear Regression Model for Stock Returns Prediction
This script implements a linear regression model to predict stock returns
using previous day's returns as the independent variable.
"""
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')


def load_and_prepare_data(file_path):
    """
    Load and preprocess the stock data.
    Args:
        file_path (str): Path to the CSV file
    Returns:
        pandas.DataFrame: Processed DataFrame
    """
    # Read CSV with semicolon separator, parse index as datetime, no header inference issues
    df = pd.read_csv(file_path, sep=';', index_col=0, parse_dates=True)
    # Rename lowercase 'close' to 'Close' to match the rest of the script
    df = df.rename(columns={'close': 'Close'})
    # Calculate returns and previous day's returns
    df['return'] = df['Close'].pct_change()
    df['prev_day_returns'] = df['return'].shift()
    df.dropna(inplace=True)
    return df


def split_data(df, test_fraction=0.2):
    """
    Split data into training and testing sets.
    Args:
        df (pandas.DataFrame): Input DataFrame
        test_fraction (float): Fraction of data for testing
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    X = df[['prev_day_returns']]
    y = df['return']
    split_index = int(len(X) * (1 - test_fraction))
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Train the linear regression model.
    Args:
        X_train (pandas.DataFrame): Training features
        y_train (pandas.Series): Training target
    Returns:
        sklearn.linear_model.LinearRegression: Trained model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(y_test, y_pred):
    """
    Calculate and print model evaluation metrics.
    Args:
        y_test (pandas.Series): Actual test values
        y_pred (numpy.ndarray): Predicted values
    """
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"R² Score: {r2:.4f}")


def plot_regression(X_test, y_test, y_pred, save_path='regression_plot.png'):
    """
    Plot the regression line and actual data points, saving the plot to a file.
    Args:
        X_test (pandas.DataFrame): Test features
        y_test (pandas.Series): Test target values
        y_pred (numpy.ndarray): Predicted values
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 7))
    plt.scatter(X_test, y_test, color='black', label='Actual')
    plt.plot(X_test, y_pred, color='blue',
             linewidth=3, label='Regression Line')
    plt.title('Linear Regression Model', fontsize=14)
    plt.xlabel('Previous Day Returns', fontsize=12)
    plt.ylabel('Current Day Returns', fontsize=12)
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory


def main():
    """Main function to run the linear regression analysis."""
    # Define data path
    # data_path = '../data/SPY.csv'
    data_path = '../data/Bitcoin_08_08_2025-09_10_2025_historical_data_coinmarketcap.csv'
    # Load and prepare data
    df = load_and_prepare_data(data_path)
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(df)
    # Train the model
    model = train_model(X_train, y_train)
    # Make predictions
    y_pred = model.predict(X_test)
    # Evaluate model
    evaluate_model(y_test, y_pred)
    # Plot results and save to file
    plot_regression(X_test, y_test, y_pred,
                    save_path='../outputs/linear_regression.png')


if __name__ == "__main__":
    main()
