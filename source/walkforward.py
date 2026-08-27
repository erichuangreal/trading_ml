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

# Walk-forward retrains as it advances, so testing earlier does not permanently
# cost training data -- only the first blocks train on less. At 5-day the sample
# went from ~82 non-overlapping periods to ~181, halving the error bar.
TEST_START = pd.Timestamp("2023-01-01")
TEST_END = pd.Timestamp("2026-08-25")

# Trading days between refits. 1 = retrain every day, 5 = once a week.
RETRAIN_EVERY = 5

# None uses the entire training dataset
# Set it to 365 to use data for the recent year only
TRAIN_WINDOW = None

# 5-day. Costs are paid once per hold instead of five times, which is what turns
# a positive gross return into a positive net one. The price is ~82
# non-overlapping scored periods against ~410 at 1-day, so the error bar widens
# from +-0.38pp to +-0.63pp -- read the sign, not the magnitude. Moving
# TEST_START back to 2023 would roughly double the sample.
TARGET = "future_return_5d"

# Classifier. Tested head to head against the regressor on the same 24 features
# and window: the edge was near identical (+0.56pp vs +0.51pp) but the monthly
# spread was 43% tighter (SD 0.0133 vs 0.0190), so t went 1.83 vs 1.32. Training
# on the ordering gives a steadier ranking, not a bigger one.
#
# This one constant also flips xgboost_pipeline's estimator, PRED_THRESHOLD, and
# whether rank_testing divides by volatility.
TRAIN_TARGET = "beats_median_5d"
HORIZON = 5

# A regressor predicts a return, so its sign is the predicted direction. A
# classifier predicts P(beats median), which lives in [0, 1] and crosses at 0.5
# -- comparing that against 0 would mark every row as "up" and collapse the
# directional edge to exactly the baseline.
IS_CLASSIFIER = TRAIN_TARGET.startswith("beats_median")
PRED_THRESHOLD = 0.5 if IS_CLASSIFIER else 0.0


def directional_accuracy(y_true, y_pred, threshold=PRED_THRESHOLD):
    direction_accuracy = (
        (y_pred > threshold) == (y_true > 0)
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


def embargoed_splits(dates, n_splits=5, horizon=HORIZON):
    """TimeSeriesSplit folds with the last `horizon` rows of each training fold
    removed.

    TimeSeriesSplit puts validation immediately after training, so with a
    forward-looking target the last few training labels span the validation
    period. Without the gap the grid search scores on partly-known answers and
    picks whatever exploits that hardest.
    """
    from sklearn.model_selection import TimeSeriesSplit

    unique_dates = np.sort(pd.unique(dates))
    folds = []

    for train_idx, test_idx in TimeSeriesSplit(n_splits=n_splits).split(unique_dates):
        embargoed = train_idx[:-horizon] if len(train_idx) > horizon else train_idx

        train_dates = set(unique_dates[embargoed])
        test_dates = set(unique_dates[test_idx])

        folds.append((
            np.flatnonzero(dates.isin(train_dates)),
            np.flatnonzero(dates.isin(test_dates)),
        ))

    return folds


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
    target=TRAIN_TARGET,
    retrain_every=RETRAIN_EVERY,
    train_window=TRAIN_WINDOW,
    test_start=TEST_START,
    test_end=TEST_END,
    horizon=HORIZON,
):

    in_test = (data["date"] >= test_start) & (data["date"] < test_end)
    test_dates = np.sort(data.loc[in_test, "date"].unique())

    # Every horizon-th day, so the target windows never overlap. Refitting still
    # walks every day -- only the days that get scored are thinned.
    scored_dates = set(test_dates[::horizon])

    blocks = [
        test_dates[i:i + retrain_every]
        for i in range(0, len(test_dates), retrain_every)
    ]

    print(f"\n{len(test_dates)} test days, {len(scored_dates)} scored "
          f"(every {horizon}), {len(blocks)} refits, "
          f"{horizon}-day embargo, "
          f"train window: {'expanding' if train_window is None else str(train_window) + ' days'}")

    # Needed to embargo in trading days rather than calendar days.
    all_dates = np.sort(data["date"].unique())

    results = []
    start_time = time.time()

    for n, block in enumerate(blocks, 1):
        cutoff = block[0]

        # `date < cutoff` is not enough once the target looks forward. A row
        # dated cutoff-1 carries a label spanning [cutoff-1, cutoff+4], so four
        # of its five days sit inside the block about to be predicted -- and the
        # label is a cross-sectional rank over that window, which is nearly the
        # question being asked. Drop the last `horizon` trading days so no
        # training label overlaps the test block at all.
        cutoff_index = int(np.searchsorted(all_dates, cutoff))
        embargo_index = max(0, cutoff_index - horizon)
        train_end = all_dates[embargo_index]

        train = data[data["date"] < train_end]
        if train_window is not None:
            train = train[train["date"] >= train_end - pd.Timedelta(days=train_window)]

        scored_in_block = [day for day in block if day in scored_dates]
        if not scored_in_block:
            continue

        test = data[data["date"].isin(scored_in_block)].copy()

        fitted = clone(model)
        fitted.fit(train[features], train[target])

        # A classifier's probability of beating the median is the ranking score.
        # It is better behaved than a regressor's raw output, whose magnitude
        # tracks volatility rather than confidence.
        if hasattr(fitted, "predict_proba"):
            test["pred"] = fitted.predict_proba(test[features])[:, 1]
        else:
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

    # R2 compares predictions to returns on the same scale. A probability is not
    # on that scale, so it would report a large meaningless negative. Rank IC
    # measures ordering and works for either.
    test_r2 = float("nan") if IS_CLASSIFIER else r2_score(y_test, y_pred)
    test_ic = rank_ic(y_test, y_pred)

    # Per-month view, so one lucky month cannot hide behind the yearly average.
    pooled = pooled.copy()
    pooled["correct"] = (y_pred > PRED_THRESHOLD) == (y_test > 0)
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
        f.write(f"Rank IC on Test Set: {test_ic:.4f}\n")
        f.write(f"Months Beaten: {months_won}/{len(monthly)}\n")
        f.write(f"\nMONTH BY MONTH:\n")
        f.write(monthly.to_string(float_format=lambda v: f"{v:.4f}") + "\n")

    print(f"\n✓ {name} walk-forward completed in {elapsed_time:.2f}s ({elapsed_time/60:.1f}m)")
    print(f"Test Days: {pooled['date'].nunique()}, Test Rows: {len(pooled)}")
    print(f"Always Up Accuracy: {baseline:.4f}")
    print(f"Directional Accuracy on Test Set: {test_acc:.4f}")
    print(f"Edge: {test_acc - baseline:+.4f}")
    print(f"R2 Score on Test Set: {test_r2:.4f}")
    print(f"Rank IC on Test Set: {test_ic:.4f}")
    print(f"Months Beaten: {months_won}/{len(monthly)}")
    print(f"\n{monthly.to_string(float_format=lambda v: f'{v:.4f}')}")
    print(f"\nSaved to {run_path}")

    return test_acc, baseline, test_r2, run_name
