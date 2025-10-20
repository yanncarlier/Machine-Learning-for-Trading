"""
    Title: Sample Support Vector Classifier Trading Strategy
    Description: This strategy will use support vector classifier to predict 
                 next day's price movement. This is a long only strategy 
                 which rebalances portfolio weights every day and retrains the model
                 at the start of each month.
    Style tags: Systematic
    Asset class: Equities
    Dataset: US Equities
    ############################# DISCLAIMER #############################
    This is a strategy template only and should not be
    used for live trading without appropriate backtesting and tweaking of
    the strategy parameters.
    ######################################################################
"""
# Import numpy
import numpy as np
# Import machine learning libraries
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# Import blueshift libraries
from blueshift.api import (
    symbol,
    order_target_percent,
    schedule_function,
    date_rules,
    time_rules,
    get_datetime
)


def initialize(context):
    # Define symbol
    context.security = symbol('AMZN')
    # Lookback to fetch data
    context.lookback = 200
    # The train-test split
    context.split_percentage = 0.8
    # The flag variable used to check whether to retrain the model or not
    context.retrain_flag = True
    # Variable to store train and test dataset
    context.X_train = None
    context.X_test = None
    context.y_train = None
    context.y_test = None
    # The variable to store the classifier
    context.cls = None
    # Schedule the retrain_model function every month
    schedule_function(
        retrain_model,
        date_rule=date_rules.month_start(),
        time_rule=time_rules.market_open()
    )
    # Schedule the rebalance function to run daily at market close
    schedule_function(
        rebalance,
        date_rule=date_rules.every_day(),
        time_rule=time_rules.market_close(minutes=1)
    )


def retrain_model(context, data):
    """
        A function to retrain the  model. This function is called by
        the schedule_function in the initialize function.
    """
    context.retrain_flag = True


def rebalance(context, data):
    try:
        Df = data.history(
            context.security,
            ['open', 'high', 'low', 'close', 'volume'],
            context.lookback,
            '1d')
    except IndexError:
        return
    # Create predictor variables
    Df['open-close'] = Df['open'] - Df['close']
    Df['high-low'] = Df['high'] - Df['low']
    Df = Df.dropna()
    # Store all predictor variables in a variable X
    X = Df[['open-close', 'high-low']]
    if context.retrain_flag:
        context.retrain_flag = False
        # Create target variable
        y = np.where(Df['close'].shift(-1) > Df['close'], 1, 0)
        # Split the data into train and test
        split = int(context.split_percentage*len(Df))
        # Train data set
        context.X_train = X[:split]
        context.y_train = y[:split]
        # Test data set
        context.X_test = X[split:]
        context.y_test = y[split:]
        # Support vector classifier
        context.cls = SVC().fit(context.X_train, context.y_train)
    # Test accuracy
    accuracy_test = accuracy_score(context.y_test,
                                   context.cls.predict(context.X_test))
    # Predicted Signal
    predicted_signal = context.cls.predict(X)[-1]
    print("{} Accuracy test: {}, Prediction: {}".format(
        get_datetime(), accuracy_test, predicted_signal))
    # Place the orders
    if accuracy_test > 0.5:
        if predicted_signal == 1:
            order_target_percent(context.security, 1)
        else:
            order_target_percent(context.security, 0)
    else:
        order_target_percent(context.security, 0)
