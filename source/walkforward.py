import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.base import clone
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from joblib import dump
import time

PROCESSED_PATH = Path("data/processed_data")
MODELS_PATH = Path("models")

TEST_START = pd.Timestamp("2025-01-01")
TEST_END = pd.Timestamp("2026-01-01")

# Trading days between refits. 1 = retrain every day, 5 = once a week.
RETRAIN_EVERY = 5

# None uses the entire training dataset
# Set it to 365 to use data for the recent year only
TRAIN_WINDOW = None

TARGET = "future_return_1d"


def directional_accuracy(y_true, y_pred):
    direction_accuracy = (
        (y_pred > 0) == (y_true > 0)
    ).mean()
    return direction_accuracy


def always_up_accuracy(y_test):
    return (y_test > 0).mean()


def rank_ic(y_true, y_pred):
    """Spearman correlation between predicted and actual ordering.
    Uses rank N to choose the best parameters.
    """
    correlation = spearmanr(y_true, y_pred).statistic

    return 0.0 if np.isnan(correlation) else correlation


def load_data():
    data = pd.read_parquet(PROCESSED_PATH / "cleaned_output.parquet")
    data = data.sort_values(["date", "ticker"]).reset_index(drop=True)

    print(f"Loaded {len(data)} rows, "
          f"{data['date'].min().date()} to {data['date'].max().date()}, "
          f"{data['ticker'].nunique()} tickers")

    return data


def walk_forward(
    model,
    data,
    features,
    target=TARGET,
    retrain_every=RETRAIN_EVERY,
    train_window=TRAIN_WINDOW,
    test_start=TEST_START,
    test_end=TEST_END,
):

    in_test = (data["date"] >= test_start) & (data["date"] < test_end)
    test_dates = np.sort(data.loc[in_test, "date"].unique())

    blocks = [
        test_dates[i:i + retrain_every]
        for i in range(0, len(test_dates), retrain_every)
    ]

    print(f"\n{len(test_dates)} test days, {len(blocks)} refits, "
          f"train window: {'expanding' if train_window is None else str(train_window) + ' days'}")

    results = []
    start_time = time.time()

    for n, block in enumerate(blocks, 1):
        cutoff = block[0]

        # Strictly earlier than the block being predicted, so nothing leaks.
        train = data[data["date"] < cutoff]
        if train_window is not None:
            train = train[train["date"] >= cutoff - pd.Timedelta(days=train_window)]

        test = data[data["date"].isin(block)].copy()

        fitted = clone(model)
        fitted.fit(train[features], train[target])
        test["pred"] = fitted.predict(test[features])

        results.append(test)

        if n % max(1, len(blocks) // 10) == 0:
            elapsed = time.time() - start_time
            remaining = elapsed / n * (len(blocks) - n)
            print(f"  {n}/{len(blocks)} refits | {elapsed:.0f}s elapsed | ~{remaining:.0f}s left")

    print(f"Walk-forward completed in {time.time() - start_time:.0f}s")
    return pd.concat(results, ignore_index=True)


def report_walkforward(name, model_label, params, pooled, elapsed_time, final_model=None):
    """Score pooled walk-forward predictions and save them the way tuning does."""
    y_test = pooled[TARGET]
    y_pred = pooled["pred"]

    test_acc = directional_accuracy(y_test, y_pred)
    baseline = always_up_accuracy(y_test)
    test_r2 = r2_score(y_test, y_pred)

    # Per-month view, so one lucky month cannot hide behind the yearly average.
    pooled = pooled.copy()
    pooled["correct"] = (y_pred > 0) == (y_test > 0)
    pooled["up"] = y_test > 0

    monthly = pooled.groupby(pooled["date"].dt.to_period("M")).agg(
        days=("date", "nunique"),
        rows=("date", "size"),
        model=("correct", "mean"),
        always_up=("up", "mean"),
    )
    monthly["edge"] = monthly["model"] - monthly["always_up"]
    months_won = (monthly["edge"] > 0).sum()

    run_name = f"{name}_walkforward_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_path = MODELS_PATH / run_name
    run_path.mkdir(parents=True, exist_ok=True)

    # A dict means one model per seed -- save them all, since averaging their
    # predictions is what reproduces the ensemble that was scored above.
    if isinstance(final_model, dict):
        for seed, fitted in final_model.items():
            dump(fitted, run_path / f"model_seed{seed}.joblib")
    elif final_model is not None:
        dump(final_model, run_path / "model.joblib")

    pooled.to_parquet(run_path / "predictions.parquet", index=False)

    with open(run_path / "training.log", "w") as f:
        f.write(f"Run: {run_name}\n")
        f.write(f"Model: {model_label} (Walk-forward)\n")
        f.write(f"Params: {params}\n")
        f.write(f"Retrain Every: {RETRAIN_EVERY} trading day(s)\n")
        f.write(f"Train Window: {'expanding' if TRAIN_WINDOW is None else str(TRAIN_WINDOW) + ' days'}\n")
        f.write(f"Walk-forward Time: {elapsed_time:.2f}s\n")
        f.write(f"TESTING RESULTS:\n")
        f.write(f"Test Period: {pooled['date'].min().date()} to {pooled['date'].max().date()}\n")
        f.write(f"Test Days: {pooled['date'].nunique()}\n")
        f.write(f"Test Rows: {len(pooled)}\n")
        f.write(f"Always Up Accuracy: {baseline:.4f}\n")
        f.write(f"Directional Accuracy on Test Set: {test_acc:.4f}\n")
        f.write(f"Edge: {test_acc - baseline:+.4f}\n")
        f.write(f"R2 Score on Test Set: {test_r2:.4f}\n")
        f.write(f"Months Beaten: {months_won}/{len(monthly)}\n")
        f.write(f"\nMONTH BY MONTH:\n")
        f.write(monthly.to_string(float_format=lambda v: f"{v:.4f}") + "\n")

    print(f"\n✓ {name} walk-forward completed in {elapsed_time:.2f}s ({elapsed_time/60:.1f}m)")
    print(f"Test Days: {pooled['date'].nunique()}, Test Rows: {len(pooled)}")
    print(f"Always Up Accuracy: {baseline:.4f}")
    print(f"Directional Accuracy on Test Set: {test_acc:.4f}")
    print(f"Edge: {test_acc - baseline:+.4f}")
    print(f"R2 Score on Test Set: {test_r2:.4f}")
    print(f"Months Beaten: {months_won}/{len(monthly)}")
    print(f"\n{monthly.to_string(float_format=lambda v: f'{v:.4f}')}")
    print(f"\nSaved to {run_path}")

    return test_acc, baseline, test_r2, run_name
