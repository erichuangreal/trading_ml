"""
Run order: extract_data -> fundamentals -> edgar_fundamentals -> process_data ->
predict. Skipping any of the data steps scores today's prices against stale
fundamentals.

MAKE SURE TO RUN AFTER 4 pm to ensure that today's bar has closed.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load

from features import FEATURES
from walkforward import PROCESSED_PATH, MODELS_PATH, HORIZON, TRAIN_TARGET
from rank_testing import (
    inverse_vol_weights,
    EXCLUDE_NEAR_EARNINGS,
    EARNINGS_EXCLUSION_DAYS,
    VOL_TARGET,
    VOL_LOOKBACK,
    MIN_EXPOSURE,
    MAX_EXPOSURE,
    PERIODS_PER_YEAR,
    TOP_N_VALUES,
)

# Select the model you want to use: the exact run directory name under models/.
RUN_NAME = "v4xgboost_walkforward_2026-08-26_22-53-59"

# Basket size. 3 is the best model so far.
TOP_N = TOP_N_VALUES[1] if len(TOP_N_VALUES) > 1 else TOP_N_VALUES[0]

CAPITAL = 1_000

# Below this share of normal volume, the newest bar is treated as an unfinished
# session rather than a real close.
PARTIAL_BAR_VOLUME = 0.5


def current_exposure(run_dir, n=TOP_N):
    """The same 17.3%-target scalar the backtest applied, off the latest data.

    Sized on the strategy's own realised volatility rather than the market's.
    Once there is a live track record this should read from that instead -- the
    saved series stops on the last day of the test period and will drift further
    out of date every month it is not replaced.
    """
    if VOL_TARGET is None:
        return 1.0

    path = Path(run_dir) / f"rank_daily_top{n}.parquet"
    if not path.exists():
        print(f"  no {path.name} in the run -- holding unlevered")
        return 1.0

    recent = pd.read_parquet(path)["scaled_return"].tail(VOL_LOOKBACK)
    if len(recent) < 2:
        return 1.0

    realised = recent.std() * np.sqrt(PERIODS_PER_YEAR)
    if not np.isfinite(realised) or realised == 0:
        return 1.0

    return float(np.clip(VOL_TARGET / realised, MIN_EXPOSURE, MAX_EXPOSURE))


def check_bar_complete(data, as_of):
    """Warn if the newest bar looks like a session still in progress.
    Will incorrectly flag half-days or low vol days"""
    latest = data[data["date"] == as_of]
    prior = (
        data[data["date"] < as_of]
        .groupby("ticker")["volume"]
        .apply(lambda s: s.tail(20).mean())
    )

    ratio = (latest.set_index("ticker")["volume"] / prior).dropna()
    if ratio.empty:
        return float("nan")

    median_ratio = float(ratio.median())

    if median_ratio < PARTIAL_BAR_VOLUME:
        print(f"  WARNING: {as_of.date()} traded {median_ratio:.0%} of normal volume "
              f"-- this looks like an unfinished session, so the closes these "
              f"picks are built on are not closes. Re-run after 4pm ET.")

    return median_ratio


def todays_picks(run_name=RUN_NAME, n=TOP_N, capital=CAPITAL):
    run_dir = MODELS_PATH / run_name

    data = pd.read_parquet(PROCESSED_PATH / "cleaned_output.parquet")
    model = load(run_dir / "model.joblib")

    as_of = data["date"].max()
    day = data[data["date"] == as_of].copy()

    if day[TRAIN_TARGET].notna().any():
        print(
            f"  note: {as_of.date()} already has a known {HORIZON}-day outcome, so "
            f"this is a backfilled date rather than a live one. Re-run "
            f"extract_data.py and process_data.py for today's prices."
        )

    check_bar_complete(data, as_of)

    # A classifier's probability of beating the median is the ranking score --
    # the same one walk_forward scored. No division by volatility: that collapses
    # a probability clustered near 0.5 into a ranking by lowest volatility.
    day["pred"] = model.predict_proba(day[FEATURES])[:, 1]

    pool = day
    if EXCLUDE_NEAR_EARNINGS and "days_to_earnings" in day.columns:
        reporting_soon = (
            (day["days_to_earnings"] <= EARNINGS_EXCLUSION_DAYS)
            | (day["days_since_earnings"] <= EARNINGS_EXCLUSION_DAYS)
        ).fillna(False)

        tradeable = day[~reporting_soon]
        if len(tradeable) >= 2 * n:
            pool = tradeable

    top = pool.nlargest(n, "pred").copy()

    exposure = current_exposure(run_dir, n)
    top["weight"] = inverse_vol_weights(top) * exposure
    top["dollars"] = top["weight"] * capital

    return as_of, exposure, run_dir, top


def report(as_of, exposure, run_dir, top, capital=CAPITAL):
    picks = top[["ticker", "pred", "volatility_20d", "weight", "dollars"]].copy()
    picks["pred"] = picks["pred"].round(4)
    picks["volatility_20d"] = picks["volatility_20d"].round(4)
    picks["weight"] = (picks["weight"] * 100).round(1)
    picks["dollars"] = picks["dollars"].round(2)
    picks = picks.rename(columns={"weight": "weight_%"})

    invested = float(top["dollars"].sum())

    print(f"\nAs of:     {as_of.date()}   (hold {HORIZON} trading days)")
    print(f"Model:     {run_dir.name}")
    print(f"Exposure:  {exposure:.2f}x   ->   "
          f"${invested:,.2f} invested, ${capital - invested:,.2f} in cash")
    print()
    print(picks.to_string(index=False))
    print(
        "\nThe ranking edge is the measured part (+1.85pp, t = 2.19)."
    )


if __name__ == "__main__":
    report(*todays_picks(RUN_NAME))
