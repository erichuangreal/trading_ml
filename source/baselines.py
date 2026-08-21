from narwhals.selectors import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from joblib import dump
from pathlib import Path
from datetime import datetime

MODELS_PATH = Path("models")

test_data = pd.read_parquet("data/processed_data/cleaned_output.parquet")

def buy_and_hold(test_data):
    start_price = test_data["close"].iloc[0]
    end_price = test_data["close"].iloc[-1]

    return (end_price / start_price) - 1

def historic_average(test_data):
    return test_data["close"].pct_change().mean()


MODEL_PATH = Path("models")

def ridge_regression(test_data):
    features = [
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
    x = test_data[features]
    y = test_data["future_return_1d"]
    clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(x, y)
    
    run_name = f"ridge_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    run_path = MODELS_PATH / run_name
    run_path.mkdir(parents=True, exist_ok=True)

    # Save model
    dump(clf, run_path / "model.joblib")

    # Save logs
    with open(run_path / "training.log", "w") as f:
        f.write(f"Run: {run_name}\n")
        f.write(f"Alpha: {clf.alpha_}\n")
        f.write(f"R2 Score: {clf.score(x, y)}\n")
        f.write(f"Best Alpha: {clf.alpha_}\n")
        
    return clf.score(x, y), clf.alpha_

if __name__ == "__main__":
    print("Ridge Regression R2 Score:", ridge_regression(test_data))
    