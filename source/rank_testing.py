"""
Reads the pooled predictions a walk-forward run saved and asks a different
question from directional accuracy: on each day, did the model correctly sort
the tickers against each other?

Two tests:
  1. Median split -- for every ticker, did the model put it on the right side of
     that day's median return. Baseline is fixed at ~50% by construction, so it
     does not drift with the market the way always-up does.
  2. Top N -- take the model's highest-ranked tickers each day and see where they
     actually landed. Edited to rank by predicted return per unit of risk (pred / volatility_20d)
     to prevent the model from choosing high volatile stocks like SMCI or TTD.
     The influence of these tickers can be seen in the top N rankings, as most of these
     stocks are present in both the top and bottom N picks. UPDATED now.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

MODELS_PATH = Path("models")

TARGET = "future_return_1d"
TOP_N = 3


def load_predictions(run_dir=None):
    """Load predictions.parquet from a walk-forward run. Defaults to the newest."""
    if run_dir is None:
        runs = sorted(MODELS_PATH.glob("*_walkforward_*/predictions.parquet"))
        if not runs:
            raise FileNotFoundError(
                f"No walk-forward runs under {MODELS_PATH}. "
                f"Run the walk-forward block in random_forest.py first."
            )
        path = runs[-1]
    else:
        path = Path(run_dir) / "predictions.parquet"
        if not path.exists():
            raise FileNotFoundError(f"No predictions.parquet in {run_dir}")

    pooled = pd.read_parquet(path)
    print(f"Loaded {len(pooled)} predictions from {path.parent.name}")

    return pooled, path.parent.name


def add_ranks(pooled):
    """Rank predictions and actuals within each day."""
    pooled = pooled.copy()

    # pct=True puts ranks on 0-1 so the median split is just > 0.5, and the
    # comparison stays valid on days where a ticker is missing.
    pooled["actual_rank"] = pooled.groupby("date")[TARGET].rank(pct=True)
    pooled["pred_rank"] = pooled.groupby("date")["pred"].rank(pct=True)

    pooled["actual_beats"] = pooled["actual_rank"] > 0.5
    pooled["pred_beats"] = pooled["pred_rank"] > 0.5
    pooled["rank_correct"] = pooled["pred_beats"] == pooled["actual_beats"]

    return pooled


def median_test(pooled):
    """Per-day counts of how many tickers the model put on the right side."""
    daily = pooled.groupby("date").agg(
        tickers=("ticker", "size"),
        predicted_beats=("pred_beats", "sum"),
        actual_beats=("actual_beats", "sum"),
        correct=("rank_correct", "sum"),
    )
    daily["accuracy"] = daily["correct"] / daily["tickers"]

    return daily


def top_n_test(pooled, n=TOP_N):
    """Where the model's n highest-ranked tickers actually landed each day.

    Ranks on predicted return per unit of risk. Ranking on the raw prediction
    instead picks whatever is most volatile -- a regressor trained on absolute
    returns outputs bigger numbers for bigger-moving stocks, so magnitude tracks
    volatility rather than confidence. Measured at 2.1x universe volatility with
    a worse spread, so it is not worth keeping as an option.
    """
    rows = []

    for date, day in pooled.groupby("date"):
        day = day.copy()

        # Zero volatility would divide to inf and hijack nlargest.
        day["score"] = day["pred"] / day["volatility_20d"].replace(0, np.nan)

        top = day.nlargest(n, "score")
        bottom = day.nsmallest(n, "score")
        actual_top = set(day.nlargest(n, TARGET)["ticker"])

        rows.append({
            "date": date,
            "tickers": len(day),
            "picks": ", ".join(top["ticker"]),
            "mean_actual_rank": top["actual_rank"].mean(),
            "beat_median": int(top["actual_beats"].sum()),
            "in_actual_top": len(set(top["ticker"]) & actual_top),
            "top_return": top[TARGET].mean(),
            "bottom_return": bottom[TARGET].mean(),
            "universe_return": day[TARGET].mean(),
            "top_volatility": top["volatility_20d"].mean(),
            "universe_volatility": day["volatility_20d"].mean(),
        })

    daily = pd.DataFrame(rows).set_index("date")
    daily["excess_return"] = daily["top_return"] - daily["universe_return"]
    daily["spread"] = daily["top_return"] - daily["bottom_return"]

    return daily


def top_n_summary(daily, days, n, tickers_per_day):
    """Format the top-N block for the log."""
    mean_rank = daily["mean_actual_rank"].mean()
    hit_rate = daily["beat_median"].sum() / (days * n)
    excess = daily["excess_return"].mean()
    spread = daily["spread"].mean()
    overlap = daily["in_actual_top"].mean()
    overlap_chance = n * n / tickers_per_day
    vol_ratio = daily["top_volatility"].mean() / daily["universe_volatility"].mean()

    return [
        f"TOP {n} -- ranked by pred / volatility_20d",
        f"  Mean actual percentile: {mean_rank:.4f}   (0.5000 = no skill)",
        f"  Beat median:            {hit_rate:.4f}   (0.5000 = no skill)",
        f"  In actual top {n}:         {overlap:.4f}   ({overlap_chance:.4f} by chance)",
        f"  Excess return:          {excess * 10000:+.2f} bps/day vs universe",
        f"  Top-minus-bottom {n}:     {spread * 10000:+.2f} bps/day",
        f"  Volatility of picks:    {vol_ratio:.2f}x universe   (1.00 = same risk)",
    ]


def report_ranks(pooled, run_name, n=TOP_N):
    """Score both tests and write the results to a log."""
    pooled = add_ranks(pooled)

    daily_median = median_test(pooled)
    daily_top = top_n_test(pooled, n)

    days = len(daily_median)
    tickers_per_day = daily_median["tickers"].mean()

    # --- median split ---
    accuracy = pooled["rank_correct"].mean()
    baseline = pooled["actual_beats"].mean()

    # --- top N ---
    # A model with no skill lands its picks at the middle of the pack (0.5),
    # gets half of them over the median, and earns the universe average.
    top_block = top_n_summary(daily_top, days, n, tickers_per_day)

    monthly = daily_median.groupby(daily_median.index.to_period("M")).agg(
        days=("accuracy", "size"),
        accuracy=("accuracy", "mean"),
    )
    by_month = daily_top.groupby(daily_top.index.to_period("M"))

    monthly["top_rank"] = by_month["mean_actual_rank"].mean()
    monthly["excess_bps"] = by_month["excess_return"].mean() * 10000

    lines = [
        f"Run: {run_name}",
        f"Scored: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Test period: {daily_median.index.min().date()} to {daily_median.index.max().date()}",
        f"Days: {days}",
        f"Rows: {len(pooled)}",
        f"Tickers per day: {tickers_per_day:.1f}",
        "",
        "MEDIAN SPLIT",
        f"  Did the model put each ticker on the right side of the day's median?",
        f"  Rank Accuracy:  {accuracy:.4f}",
        f"  Baseline:       {baseline:.4f}",
        f"  Edge:           {accuracy - baseline:+.4f}",
        "",
        *top_block,
        "",
        "MONTH BY MONTH:",
        monthly.to_string(float_format=lambda v: f"{v:.4f}"),
        "",
        f"WORST {n} DAYS BY TOP-{n} RETURN:",
        daily_top.nsmallest(n, "top_return")[
            ["picks", "top_return", "universe_return", "mean_actual_rank"]
        ].to_string(float_format=lambda v: f"{v:.4f}"),
        "",
        f"BEST {n} DAYS BY TOP-{n} RETURN:",
        daily_top.nlargest(n, "top_return")[
            ["picks", "top_return", "universe_return", "mean_actual_rank"]
        ].to_string(float_format=lambda v: f"{v:.4f}"),
    ]

    output = "\n".join(lines)
    print("\n" + output)

    run_path = MODELS_PATH / run_name
    run_path.mkdir(parents=True, exist_ok=True)

    with open(run_path / "rank_testing.log", "w") as f:
        f.write(output + "\n")

    daily_median.to_parquet(run_path / "rank_daily_median.parquet")
    daily_top.to_parquet(run_path / "rank_daily_top.parquet")

    print(f"\nSaved to {run_path / 'rank_testing.log'}")

    return daily_median, daily_top


if __name__ == "__main__":
    pooled, run_name = load_predictions()
    report_ranks(pooled, run_name)
