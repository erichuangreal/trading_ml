import pandas as pd
from sklearn.linear_model import RidgeCV
from pathlib import Path
from joblib import load

test_data = pd.read_parquet("data/processed_data/cleaned_test_output.parquet")

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
y_test = test_data["future_return_1d"]

def test_model(model_name, test_data, features, y_test):
    MODEL_PATH = Path("models") / model_name / "model.joblib"
    clf = load(MODEL_PATH)
    
    predictions = clf.predict(test_data[features])

    direction_accuracy = (
        (predictions > 0) == (y_test > 0)
    ).mean()
    print("Directional accuracy for model: ", model_name, direction_accuracy)


always_up_accuracy = (y_test > 0).mean()
print("Always up accuracy:", always_up_accuracy)

test_model("ridge_2026-08-21_18-25-17", test_data, features, y_test)
test_model("random_forest_tuned_2026-08-22_02-15-15", test_data, features, y_test)
test_model("xgboost_tuned_2026-08-22_02-15-17", test_data, features, y_test)