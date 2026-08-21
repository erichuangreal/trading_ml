import pandas as pd
from sklearn.linear_model import RidgeCV
from pathlib import Path
from joblib import load

test_data = pd.read_parquet("data/processed_data/cleaned_output.parquet")

MODEL_PATH = Path("models/ridge_2026-08-21_18-25-17/model.joblib")

clf = load(MODEL_PATH)

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
y_test = test_data["future_return_5d"]

predictions = clf.predict(test_data[features])

direction_accuracy = (
    (predictions > 0) == (y_test > 0)
).mean()

always_up_accuracy = (y_test > 0).mean()

print("Directional accuracy:", direction_accuracy)
print("Always up accuracy:", always_up_accuracy)