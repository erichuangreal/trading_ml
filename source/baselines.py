import pandas as pd
import numpy as np

test_data = pd.read_parquet("data/processed_data/cleaned_output.parquet")

def buy_and_hold(test_data):
    start_price = test_data["Close"].iloc[0]
    end_price = test_data["Close"].iloc[-1]

    return (end_price / start_price) - 1

def historic_average(test_data):
    return test_data["Close"].pct_change().mean()

def lin_reg(test_data):
    X = np.arange(len(test_data)).reshape(-1, 1)
    y = test_data["Close"].values

    # Fit a linear regression model
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X, y)

    # Predict the next value
    next_index = np.array([[len(test_data)]])
    predicted_price = model.predict(next_index)[0]

    start_price = test_data["Close"].iloc[0]
    
    return (predicted_price / start_price) - 1