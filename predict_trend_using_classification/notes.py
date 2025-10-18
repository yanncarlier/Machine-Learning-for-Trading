# Import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')
# The data is stored in the directory 'data'
# path = '../data/'
# File path
file_path = './data/Bitcoin_08_08_2025-09_10_2025_historical_data_coinmarketcap.csv'
# Read the csv file using read_csv method of pandas
# Key fixes: sep=';' for semicolon delimiter, encoding='utf-8-sig' to handle BOM
Df = pd.read_csv(file_path, sep=';', encoding='utf-8-sig', index_col=0)
# DIAGNOSTIC: Print exact column names to verify (should now work)
print("Exact column names:", Df.columns.tolist())
print("DataFrame shape:", Df.shape)
# Strip any whitespace from column names (just in case)
Df.columns = Df.columns.str.strip()
print("Stripped column names:", Df.columns.tolist())
# Column names are lowercase: 'open', 'high', 'low', 'close'
# Create predictor variables
Df['Open-Close'] = Df['open'] - Df['close']
Df['High-Low'] = Df['high'] - Df['low']
X = Df[['Open-Close', 'High-Low']]
# Target variables
y = np.where(Df['close'].shift(-1) > Df['close'], 1, -1)
# Drop rows with NaN in y (last row due to shift)
mask = ~np.isnan(y)
Df = Df[mask].copy()
y = y[mask]
X = X[mask]
print(f"Data after cleaning: {len(Df)} rows")
if len(Df) < 10:
    print("ERROR: Too few rows for training. Note: Data appears to be from Oct 2025, but filename suggests Aug-Sep—re-download if needed.")
else:
    # Split the data into train and test dataset
    split_percentage = 0.8
    split = int(split_percentage * len(Df))
    # Train data set
    X_train = X[:split]
    y_train = y[:split]
    # Test data set
    X_test = X[split:]
    y_test = y[split:]
    # Create support classifier model
    clf = SVC().fit(X_train, y_train)
    # Predicted Signal
    Df['Predicted_Signal'] = clf.predict(X)
    print(Df.tail())
    # Optional: Print accuracy on test set
    print("Test Accuracy:", accuracy_score(y_test, clf.predict(X_test)))
