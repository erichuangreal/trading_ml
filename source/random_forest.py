import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.base import clone
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import GridSearchCV
from joblib import dump
from pathlib import Path
from datetime import datetime
import time

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from walkforward import (
    walk_forward,
    report_walkforward,
    load_data,
    directional_accuracy,
    always_up_accuracy,
    rank_ic,
    embargoed_splits,
    TARGET,
    TRAIN_TARGET,
)
from rank_testing import report_ranks
from features import FEATURES

MODELS_PATH = Path("models")

train_data = pd.read_parquet("data/processed_data/cleaned_train_output.parquet")
test_data = pd.read_parquet("data/processed_data/cleaned_test_output.parquet")

train_data = train_data.sort_values(["date", "ticker"]).reset_index(drop=True)
test_data = test_data.sort_values(["date", "ticker"]).reset_index(drop=True)

RF_BEST = {
    "n_estimators": 200,
    "max_depth": 10,
}

XGB_BEST = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
}

def random_forest_pipeline(n_estimators=200, max_depth=5, random_state=42,
                           max_features="sqrt", min_samples_leaf=50):
    """The RandomForest pipeline, unfitted. Shared by training and walk-forward."""
    return make_pipeline(
        StandardScaler(),
        RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1
        )
    )


def random_forest(n_estimators=200, max_depth=5, random_state=42):
    x = train_data[FEATURES]
    y = train_data[TRAIN_TARGET]

    pipeline = random_forest_pipeline(n_estimators, max_depth, random_state)

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

def xgboost_pipeline(n_estimators=50, max_depth=7, learning_rate=0.01, random_state=42):
    """The ranking score is calculated as the probability of beating the day's median."""
    estimator = XGBClassifier if TRAIN_TARGET.startswith("beats_median") else XGBRegressor

    return make_pipeline(
        StandardScaler(),
        estimator(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0
        )
    )


def xgboost_model(n_estimators=50, max_depth=7, learning_rate=0.01, random_state=42):
    x = train_data[FEATURES]
    y = train_data[TRAIN_TARGET]

    pipeline = xgboost_pipeline(n_estimators, max_depth, learning_rate, random_state)

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


def tune_random_forest(train_data, test_data):
    """Grid search to find optimal RandomForest hyperparameters."""
    x = train_data[FEATURES]
    y = train_data[TRAIN_TARGET]
    x_test = test_data[FEATURES]
    y_test = test_data[TARGET]

    # Same factory the walk-forward uses, so max_features and min_samples_leaf
    # cannot drift apart from what the tuned params get applied to.
    pipeline = random_forest_pipeline()
    prefix = pipeline.steps[-1][0]

    param_grid = {
        f'{prefix}__n_estimators': [100, 200],
        f'{prefix}__max_depth': [5, 10],
    }

    print("Tuning RandomForest (4 combinations, 5-fold CV)...")
    start_time = time.time()
    ic_scorer = make_scorer(rank_ic)
    grid_search = GridSearchCV(pipeline, param_grid,
                               cv=embargoed_splits(train_data["date"]),
                               scoring=ic_scorer, n_jobs=-1, verbose=1)
    grid_search.fit(x, y)
    
    elapsed_time = time.time() - start_time
    
    best_model = grid_search.best_estimator_

    # A classifier's class labels are not a ranking -- its probability is. And
    # its .score() is accuracy, which cannot be computed against continuous
    # returns at all.
    if hasattr(best_model, "predict_proba"):
        y_pred = best_model.predict_proba(x_test)[:, 1]
        test_r2 = float("nan")
    else:
        y_pred = best_model.predict(x_test)
        test_r2 = best_model.score(x_test, y_test)

    test_acc = directional_accuracy(y_test, y_pred)
    test_ic = rank_ic(y_test, y_pred)

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
        f.write(f"TESTING RESULTS:\n")
        f.write(f"Always Up Accuracy: {always_up_accuracy(y_test):.4f}\n")
        f.write(f"Directional Accuracy on Test Set: {test_acc:.4f}\n")
        f.write(f"R2 Score on Test Set: {test_r2:.4f}\n")
        f.write(f"Rank IC on Test Set: {test_ic:.4f}\n")
        
    print(f"Always Up Accuracy: {always_up_accuracy(y_test):.4f}")
    print(f"Directional Accuracy on Test Set: {test_acc:.4f}")
    print(f"R2 Score on Test Set: {test_r2:.4f}")
    print(f"Rank IC on Test Set: {test_ic:.4f}")
    return grid_search


def tune_xgboost(train_data, test_data):
    """Grid search to find optimal XGBoost hyperparameters."""
    x = train_data[FEATURES]
    y = train_data[TRAIN_TARGET]
    x_test = test_data[FEATURES]
    y_test = test_data[TARGET]

    # Built by the same factory the walk-forward uses, so the estimator being
    # tuned is the one that gets run. Constructing it inline meant tuning an
    # XGBRegressor and then walking forward with an XGBClassifier.
    pipeline = xgboost_pipeline()

    # make_pipeline names the step after the class, so a classifier changes the
    # prefix from xgbregressor__ to xgbclassifier__.
    prefix = pipeline.steps[-1][0]

    param_grid = {
        f'{prefix}__n_estimators': [50, 100, 200],
        f'{prefix}__max_depth': [3, 5, 7],
        f'{prefix}__learning_rate': [0.01, 0.1, 0.3],
    }

    print("Tuning XGBoost (27 combinations, 5-fold CV)...")
    start_time = time.time()
    ic_scorer = make_scorer(rank_ic)
    grid_search = GridSearchCV(pipeline, param_grid,
                               cv=embargoed_splits(train_data["date"]),
                               scoring=ic_scorer, n_jobs=-1, verbose=1)
    grid_search.fit(x, y)

    elapsed_time = time.time() - start_time

    best_model = grid_search.best_estimator_

    # A classifier's class labels are not a ranking -- its probability is. And
    # its .score() is accuracy, which cannot be computed against continuous
    # returns at all.
    if hasattr(best_model, "predict_proba"):
        y_pred = best_model.predict_proba(x_test)[:, 1]
        test_r2 = float("nan")
    else:
        y_pred = best_model.predict(x_test)
        test_r2 = best_model.score(x_test, y_test)

    test_acc = directional_accuracy(y_test, y_pred)
    test_ic = rank_ic(y_test, y_pred)

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
        f.write(f"TESTING RESULTS:\n")
        f.write(f"Always Up Accuracy: {always_up_accuracy(y_test):.4f}\n")
        f.write(f"Directional Accuracy on Test Set: {test_acc:.4f}\n")
        f.write(f"R2 Score on Test Set: {test_r2:.4f}\n")
        f.write(f"Rank IC on Test Set: {test_ic:.4f}\n")
        
    print(f"Always Up Accuracy: {always_up_accuracy(y_test):.4f}")
    print(f"Directional Accuracy on Test Set: {test_acc:.4f}")
    print(f"R2 Score on Test Set: {test_r2:.4f}")
    print(f"Rank IC on Test Set: {test_ic:.4f}")
    return grid_search


if __name__ == "__main__":
    RUN_BASELINE = False
    RUN_TUNING = True
    RUN_WALKFORWARD = False

    if RUN_BASELINE:
        print("RandomForest R2 Score:", random_forest()[0])
        print("XGBoost R2 Score:", xgboost_model()[0])

    if RUN_TUNING:
        print("Tuning to find the best parameters...")
        # tune_random_forest(train_data, test_data)
        tune_xgboost(train_data, test_data)

    if RUN_WALKFORWARD:
        # Uses the tuned params above. Refits each block, never re-tunes.
        data = load_data()

        models = [
            # random forest is now discontinued
            #("random_forest", "RandomForestRegressor", random_forest_pipeline(**RF_BEST), RF_BEST),
            ("xgboost", "XGBRegressor", xgboost_pipeline(**XGB_BEST), XGB_BEST),
        ]

        for name, model_label, model, params in models:
            start_time = time.time()
            pooled = walk_forward(model, data, FEATURES)
            elapsed_time = time.time() - start_time

            # The model you would actually deploy: same params, fit on everything.
            final_model = clone(model).fit(data[FEATURES], data[TRAIN_TARGET])

            test_acc, baseline, test_r2, run_name = report_walkforward(
                name, model_label, params, pooled, elapsed_time, final_model
            )

            # Cross-sectional scoring, written into the same run directory.
            report_ranks(pooled, run_name)
