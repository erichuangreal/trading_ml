import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from joblib import dump
from pathlib import Path
from datetime import datetime
import time

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODELS_PATH = Path("models")

test_data = pd.read_parquet("data/processed_data/cleaned_output.parquet")

FEATURES = [
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


def random_forest(test_data, n_estimators=200, max_depth=5, random_state=42):
    """Train RandomForest model to predict 5-day future returns."""
    x = test_data[FEATURES]
    y = test_data["future_return_5d"]

    pipeline = make_pipeline(
        StandardScaler(),
        RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
    )

    print("Training RandomForest...")
    start_time = time.time()
    clf = pipeline.fit(x, y)
    r2 = clf.score(x, y)
    elapsed_time = time.time() - start_time
    print(f"✓ RandomForest training completed in {elapsed_time:.2f}s")

    run_name = f"random_forest_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_path = MODELS_PATH / run_name
    run_path.mkdir(parents=True, exist_ok=True)

    dump(clf, run_path / "model.joblib")

    with open(run_path / "training.log", "w") as f:
        f.write(f"Run: {run_name}\n")
        f.write(f"Model: RandomForestRegressor\n")
        f.write(f"N Estimators: {n_estimators}\n")
        f.write(f"Max Depth: {max_depth}\n")
        f.write(f"R2 Score: {r2}\n")
        f.write(f"Seed: {random_state}\n")
        f.write(f"Training Time: {elapsed_time:.2f}s\n")

    return r2, clf


def xgboost_model(test_data, n_estimators=50, max_depth=7, learning_rate=0.01, random_state=42):
    """Train XGBoost model to predict 5-day future returns."""
    x = test_data[FEATURES]
    y = test_data["future_return_5d"]

    pipeline = make_pipeline(
        StandardScaler(),
        XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0
        )
    )

    print("Training XGBoost...")
    start_time = time.time()
    clf = pipeline.fit(x, y)
    r2 = clf.score(x, y)
    elapsed_time = time.time() - start_time
    print(f"✓ XGBoost training completed in {elapsed_time:.2f}s")

    run_name = f"xgboost_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_path = MODELS_PATH / run_name
    run_path.mkdir(parents=True, exist_ok=True)

    dump(clf, run_path / "model.joblib")

    with open(run_path / "training.log", "w") as f:
        f.write(f"Run: {run_name}\n")
        f.write(f"Model: XGBRegressor\n")
        f.write(f"N Estimators: {n_estimators}\n")
        f.write(f"Max Depth: {max_depth}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"R2 Score: {r2}\n")
        f.write(f"Seed: {random_state}\n")
        f.write(f"Training Time: {elapsed_time:.2f}s\n")
    return r2, clf


def tune_random_forest(test_data):
    """Grid search to find optimal RandomForest hyperparameters."""
    x = test_data[FEATURES]
    y = test_data["future_return_5d"]

    pipeline = make_pipeline(
        StandardScaler(),
        RandomForestRegressor(random_state=42, n_jobs=-1)
    )

    param_grid = {
        'randomforestregressor__n_estimators': [50, 100, 200],
        'randomforestregressor__max_depth': [5, 10, 15, 20],
    }

    print("Tuning RandomForest (24 combinations, 5-fold CV)...")
    start_time = time.time()
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(x, y)
    elapsed_time = time.time() - start_time

    print(f"\n✓ RandomForest tuning completed in {elapsed_time:.2f}s ({elapsed_time/60:.1f}m)")
    print("Best RandomForest params:", grid_search.best_params_)
    print("Best CV score:", grid_search.best_score_)

    run_name = f"random_forest_tuned_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_path = MODELS_PATH / run_name
    run_path.mkdir(parents=True, exist_ok=True)

    dump(grid_search.best_estimator_, run_path / "model.joblib")

    with open(run_path / "training.log", "w") as f:
        f.write(f"Run: {run_name}\n")
        f.write(f"Model: RandomForestRegressor (Tuned)\n")
        f.write(f"Best Params: {grid_search.best_params_}\n")
        f.write(f"Best CV Score: {grid_search.best_score_}\n")
        f.write(f"Tuning Time: {elapsed_time:.2f}s\n")

    return grid_search


def tune_xgboost(test_data):
    """Grid search to find optimal XGBoost hyperparameters."""
    x = test_data[FEATURES]
    y = test_data["future_return_5d"]

    pipeline = make_pipeline(
        StandardScaler(),
        XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
    )

    param_grid = {
        'xgbregressor__n_estimators': [50, 100, 200],
        'xgbregressor__max_depth': [3, 5, 7],
        'xgbregressor__learning_rate': [0.01, 0.1, 0.3],
    }

    print("Tuning XGBoost (27 combinations, 5-fold CV)...")
    start_time = time.time()
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(x, y)
    elapsed_time = time.time() - start_time

    print(f"\n✓ XGBoost tuning completed in {elapsed_time:.2f}s ({elapsed_time/60:.1f}m)")
    print("Best XGBoost params:", grid_search.best_params_)
    print("Best CV score:", grid_search.best_score_)

    run_name = f"xgboost_tuned_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_path = MODELS_PATH / run_name
    run_path.mkdir(parents=True, exist_ok=True)

    dump(grid_search.best_estimator_, run_path / "model.joblib")

    with open(run_path / "training.log", "w") as f:
        f.write(f"Run: {run_name}\n")
        f.write(f"Model: XGBRegressor (Tuned)\n")
        f.write(f"Best Params: {grid_search.best_params_}\n")
        f.write(f"Best CV Score: {grid_search.best_score_}\n")
        f.write(f"Tuning Time: {elapsed_time:.2f}s\n")

    return grid_search


if __name__ == "__main__":
    print("RandomForest R2 Score:", random_forest(test_data)[0])
    print("XGBoost R2 Score:", xgboost_model(test_data)[0])

    # Testing best parameters found from grid search
    # tune_random_forest(test_data)
    # tune_xgboost(test_data)