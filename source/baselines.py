from narwhals.selectors import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from joblib import dump
from pathlib import Path
from datetime import datetime

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
    y = test_data["future_return_5d"]
    pipeline = make_pipeline(StandardScaler(), RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10, 100]))
    clf = pipeline.fit(x, y)
    r2 = clf.score(x, y)
    alpha = clf.named_steps["ridgecv"].alpha_
    
    run_name = f"ridge_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    run_path = MODELS_PATH / run_name
    run_path.mkdir(parents=True, exist_ok=True)

    # Save model
    dump(clf, run_path / "model.joblib")

    # Save logs
    with open(run_path / "training.log", "w") as f:
        f.write(f"Run: {run_name}\n")
        f.write(f"Alpha: {alpha}\n")
        f.write(f"R2 Score: {r2}\n")
        f.write(f"Best Alpha: {alpha}\n")
        
    return r2, alpha

if __name__ == "__main__":
    print("Ridge Regression R2 Score:", ridge_regression(test_data))
    