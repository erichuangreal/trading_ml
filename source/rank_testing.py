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

from walkforward import IS_CLASSIFIER

MODELS_PATH = Path("models")

TARGET = "future_return_5d"

# Trading days each position is held. Must match walkforward.HORIZON -- the
# scored days are spaced this far apart, so each row is one full holding period.
HORIZON = 5

# Scored at several basket sizes because they answer different questions. 3 is
# what is realistic to actually hold, but with only 3 names a day the spread's
# standard error swamps its mean and no result is readable. 15 diversifies away
# the single-name noise and shows whether the ranking works at all.
# The first entry drives the monthly table and the best/worst day listings.
TOP_N_VALUES = [1, 3, 15]
TOP_N = TOP_N_VALUES[0]

# Whether to drop names reporting within EARNINGS_EXCLUSION_DAYS from the pick
# pool. Affects the top-N block only -- the median split scores every row.
#
# Computed here from days_to/since_earnings rather than read from the parquet's
# near_earnings column, so the window can be changed without reprocessing.
EXCLUDE_NEAR_EARNINGS = True
EARNINGS_EXCLUSION_DAYS = 5

TRADING_DAYS = 252

# Long the top N and nothing else. The short leg was inverted -- bottom picks
# outran top picks while both beat the universe -- so it was subtracting from a
# working long book.
LONG_ONLY = True

# The investable benchmark. The universe average is equal-weighted over the 2026
# index members held back to 2023, so it is a hindsight portfolio; SPY is what
# someone could actually have bought over the same period.
BENCHMARK_COLUMN = f"spy_future_return_{HORIZON}d"

# Round-trip cost of turning the whole book over once, as a fraction of capital.
# 15 bps is mid-range for liquid US large caps once spread, commission and
# slippage are counted. Proportional to dollars traded, not to the number of
# names -- splitting the same capital across 15 tickers means 15 positions of
# 1/15th the size, so the total stays 15 bps.
ROUND_TRIP_COST = 0.0015

# Periods per year, and so the annualisation factor. One position per horizon.
PERIODS_PER_YEAR = TRADING_DAYS / HORIZON


# Long-only puts all capital on one side. A long/short book splits it, so its
# return is half the top-minus-bottom spread.
CAPITAL_PER_SIDE = 1.0 if LONG_ONLY else 0.5


def sharpe(period_returns, cost=0.0):
    """Annualised return per unit of risk, optionally net of a per-period cost.

    Each row is one holding period, not one day, so this annualises by the number
    of periods in a year rather than by 252.
    """
    excess = period_returns - cost

    if len(excess) < 2 or excess.std() == 0:
        return float("nan")

    return excess.mean() / excess.std() * np.sqrt(PERIODS_PER_YEAR)


def period_return(spread, cost=0.0):
    """Return on total capital for one holding period, as a fraction.

    The spread is a return per side. Splitting capital evenly between the long
    and short leg halves it.
    """
    return (spread - cost) * CAPITAL_PER_SIDE


def annualised_return(spread, cost=0.0):
    """Compounded over a year of holding periods, as a fraction."""
    per_period = period_return(spread, cost)

    # Below -100% per period there is nothing left to compound.
    if per_period <= -1:
        return -1.0

    return (1 + per_period) ** PERIODS_PER_YEAR - 1


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

        # A classifier's pred is already a volatility-free confidence, clustered
        # near 0.5. Dividing that by volatility leaves a nearly constant
        # numerator, so the ranking collapses to "lowest volatility first" --
        # picks came out at 0.40x universe volatility and hit the actual top 3
        # eight times less often than chance.
        #
        # A regressor's pred is a return, whose magnitude does track volatility,
        # so there the division is the correction it was meant to be.
        if IS_CLASSIFIER:
            day["score"] = day["pred"]
        else:
            # Zero volatility would divide to inf and hijack nlargest.
            day["score"] = day["pred"] / day["volatility_20d"].replace(0, np.nan)

        # An earnings-day move is dominated by the surprise, which none of the
        # features can see, so holding those names should add variance without
        # edge. Excluded names stay in the universe average -- only the pool the
        # picks are drawn from shrinks.
        #
        # Set False to score without it.
        if EXCLUDE_NEAR_EARNINGS and "days_to_earnings" in day.columns:
            reporting_soon = (
                (day["days_to_earnings"] <= EARNINGS_EXCLUSION_DAYS)
                | (day["days_since_earnings"] <= EARNINGS_EXCLUSION_DAYS)
            ).fillna(False)

            tradeable = day[~reporting_soon]
            if len(tradeable) >= 2 * n:
                day = tradeable

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
            # Identical across tickers on a date, so any row carries it.
            "benchmark_return": (
                day[BENCHMARK_COLUMN].iloc[0]
                if BENCHMARK_COLUMN in day.columns else np.nan
            ),
        })

    daily = pd.DataFrame(rows).set_index("date")
    daily["excess_return"] = daily["top_return"] - daily["universe_return"]
    daily["spread"] = daily["top_return"] - daily["bottom_return"]
    daily["excess_vs_benchmark"] = daily["top_return"] - daily["benchmark_return"]

    return daily


def top_n_summary(daily, days, n, tickers_per_day):
    """Format the top-N block for the log."""
    mean_rank = daily["mean_actual_rank"].mean()
    hit_rate = daily["beat_median"].sum() / (days * n)
    spread = daily["spread"].mean()
    overlap = daily["in_actual_top"].mean()
    overlap_chance = n * n / tickers_per_day
    vol_ratio = daily["top_volatility"].mean() / daily["universe_volatility"].mean()

    # Two benchmarks answering different questions. SPY is the decision -- is
    # this worth running instead of an index fund. The universe is the diagnosis
    # -- does the ranking add anything, since both use the same 90 tickers over
    # the same periods so the survivorship bias cancels.
    universe = daily["universe_return"].mean()
    excess = daily["excess_return"].mean()
    benchmark = daily["benchmark_return"].mean()
    excess_benchmark = daily["excess_vs_benchmark"].mean()

    # Long-only earns the basket's own return, and beating SPY is the bar it has
    # to clear. Long/short earns the spread, which nets the market out already.
    returns = daily["top_return"] if LONG_ONLY else daily["spread"]

    gross_sharpe = sharpe(returns)
    net_sharpe = sharpe(returns, ROUND_TRIP_COST)

    header = [
        f"TOP {n} -- ranked by {'pred' if IS_CLASSIFIER else 'pred / volatility_20d'}",
        f"  Mean actual percentile: {mean_rank:.4f}   (0.5000 = no skill)",
        f"  Beat median:            {hit_rate:.4f}   (0.5000 = no skill)",
        f"  In actual top {n}:         {overlap:.4f}   ({overlap_chance:.4f} by chance)",
        f"  Excess return:          {excess_benchmark * 10000:+.2f} bps/period vs SPY",
        f"  Top-minus-bottom {n}:     {spread * 10000:+.2f} bps/period",
        f"  Volatility of picks:    {vol_ratio:.2f}x universe   (1.00 = same risk)",
        "",
    ]

    if not LONG_ONLY:
        return header + [
            f"  LONG/SHORT TOP-BOTTOM {n}"
            f"  ({CAPITAL_PER_SIDE:.0%} long / {CAPITAL_PER_SIDE:.0%} short, rebalanced every {HORIZON}d)",
            f"    Spread per {HORIZON}d:   {spread * 10000:+.2f} bps",
            f"    Volatility:       {daily['spread'].std() * 10000:.2f} bps",
            f"    Cost assumption:  {ROUND_TRIP_COST * 10000:.1f} bps per {HORIZON}d round trip",
            "",
            f"    GROSS   {HORIZON}d return: {period_return(spread):+.4%}"
            f"   |   yearly: {annualised_return(spread):+.2%}",
            f"    NET     {HORIZON}d return: {period_return(spread, ROUND_TRIP_COST):+.4%}"
            f"   |   yearly: {annualised_return(spread, ROUND_TRIP_COST):+.2%}",
            "",
            f"    Gross Sharpe:     {gross_sharpe:+.2f}   (1.0 = good fund, 2.0 = excellent)",
            f"    Net Sharpe:       {net_sharpe:+.2f}   <- the number that decides if this is tradeable",
        ]

    # What the basket itself earned, and what holding all 90 would have earned
    # over the same periods. The gap is the part attributable to picking.
    basket = returns.mean()

    lines = header + [
        f"  LONG ONLY TOP {n}  (100% long, rebalanced every {HORIZON}d)",
        f"    Return per {HORIZON}d:   {basket * 10000:+.2f} bps",
        f"    Volatility:       {returns.std() * 10000:.2f} bps",
        f"    Cost assumption:  {ROUND_TRIP_COST * 10000:.1f} bps per {HORIZON}d round trip",
        "",
        f"    GROSS   {HORIZON}d return: {period_return(basket):+.4%}"
        f"   |   yearly: {annualised_return(basket):+.2%}",
        f"    NET     {HORIZON}d return: {period_return(basket, ROUND_TRIP_COST):+.4%}"
        f"   |   yearly: {annualised_return(basket, ROUND_TRIP_COST):+.2%}",
    ]

    if not np.isnan(benchmark):
        lines += [
            "",
            f"    vs SPY (buy and hold)  -- is this worth running at all",
            f"      SPY return:     {period_return(benchmark):+.4%}"
            f"   |   yearly: {annualised_return(benchmark):+.2%}",
            f"      NET excess:     {period_return(excess_benchmark, ROUND_TRIP_COST):+.4%}"
            f"   |   yearly: {annualised_return(excess_benchmark, ROUND_TRIP_COST):+.2%}"
            f"   <- beats SPY if positive",
        ]

    lines += [
        "",
        f"    vs UNIVERSE (equal-weight 90)  -- does the ranking add anything",
        f"      universe return: {period_return(universe):+.4%}"
        f"   |   yearly: {annualised_return(universe):+.2%}",
        f"      NET excess:      {period_return(excess, ROUND_TRIP_COST):+.4%}"
        f"   |   yearly: {annualised_return(excess, ROUND_TRIP_COST):+.2%}",
    ]

    return lines + [
        "",
        f"    Gross Sharpe:     {gross_sharpe:+.2f}   (1.0 = good fund, 2.0 = excellent)",
        f"    Net Sharpe:       {net_sharpe:+.2f}   <- the number that decides if this is tradeable",
    ]


def report_ranks(pooled, run_name, n_values=TOP_N_VALUES):
    """Score both tests and write the results to a log."""
    pooled = add_ranks(pooled)

    daily_median = median_test(pooled)

    days = len(daily_median)
    tickers_per_day = daily_median["tickers"].mean()

    # --- median split ---
    accuracy = pooled["rank_correct"].mean()
    baseline = pooled["actual_beats"].mean()

    # --- top N, at each basket size ---
    # A model with no skill lands its picks at the middle of the pack (0.5),
    # gets half of them over the median, and earns the universe average.
    daily_by_n = {size: top_n_test(pooled, size) for size in n_values}

    top_block = []
    for size in n_values:
        top_block += top_n_summary(daily_by_n[size], days, size, tickers_per_day)
        top_block.append("")

    # The first size drives the monthly table and the day listings.
    n = n_values[0]
    daily_top = daily_by_n[n]

    monthly = daily_median.groupby(daily_median.index.to_period("M")).agg(
        days=("accuracy", "size"),
        accuracy=("accuracy", "mean"),
    )
    by_month = daily_top.groupby(daily_top.index.to_period("M"))

    monthly["top_rank"] = by_month["mean_actual_rank"].mean()
    monthly["excess_bps"] = by_month["excess_vs_benchmark"].mean() * 10000
    monthly["spread_bps"] = by_month["spread"].mean() * 10000
    monthly["net_sharpe"] = by_month["spread"].apply(lambda s: sharpe(s, ROUND_TRIP_COST))

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
        f"MONTH BY MONTH (top {n}):",
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
    for size, frame in daily_by_n.items():
        frame.to_parquet(run_path / f"rank_daily_top{size}.parquet")

    print(f"\nSaved to {run_path / 'rank_testing.log'}")

    return daily_median, daily_by_n


if __name__ == "__main__":
    pooled, run_name = load_predictions()
    report_ranks(pooled, run_name)
