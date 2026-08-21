import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV

test_data = pd.read_parquet("data/processed_data/cleaned_output.parquet")

def buy_and_hold(test_data):
    start_price = test_data["Close"].iloc[0]
    end_price = test_data["Close"].iloc[-1]

    return (end_price / start_price) - 1

def historic_average(test_data):
    return test_data["Close"].pct_change().mean()


def ridge_regression(test_data):
    x = test_data[
        "close_vs_ema20",
        "close_vs_ema50",
        "close_vs_ema100",
        "return_5d",
        "return_10d",
        "return_20d",
        "volatility_5d",
        "volatility_20d",
        "rsi"
    ]
    y = 
    clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(x, y)
    clf.score(x, y) # r2 score
    