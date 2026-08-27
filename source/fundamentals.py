"""Fundamental features from yfinance.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
import time

RAW_PATH = Path("data/raw_data")

# Must match what ticker_statements fetches. income_stmt is annual; switching to
# quarterly means quarterly_income_stmt there and a shift of 4 in the growth
# calculation -- and yfinance only served 7 quarters, which was not enough for
# any TTM window.
STATEMENT_FREQUENCY = "annual"

# Days between period end and the filing being public. 10-Q is due in 40 days,
# 10-K in 60-90. Both rounded up to stay conservative.
REPORTING_LAG_DAYS = 45 if STATEMENT_FREQUENCY == "quarterly" else 90

# Periods back for the year-ago comparison: 4 quarters, or 1 year.
YEAR_AGO_PERIODS = 4 if STATEMENT_FREQUENCY == "quarterly" else 1

# Positions are excluded this many days either side of a report. Sized to the
# holding period: a position entered today is held 5 days, so a report anywhere
# in that window lands inside the trade. The move on an earnings day is dominated
# by the surprise, which none of these features see.
EARNINGS_EXCLUSION_DAYS = 5

FUNDAMENTAL_FEATURES = [
    "days_since_earnings",
    "days_to_earnings",
]

ROW_CANDIDATES = {
    "revenue": ["Total Revenue", "Operating Revenue"],
    "net_income": [
        "Net Income",
        "Net Income Common Stockholders",
        "Net Income From Continuing Operation Net Minority Interest",
    ],
    "eps": ["Diluted EPS", "Basic EPS"],
}


def pick_row(statement, key):
    """First matching row from a yfinance statement, or None."""
    if statement is None or statement.empty:
        return None

    for name in ROW_CANDIDATES[key]:
        if name in statement.index:
            return pd.to_numeric(statement.loc[name], errors="coerce")

    return None


def ticker_statements(ticker_handle, ticker):
    """Annual revenue, net income and EPS, oldest first."""
    try:
        income = ticker_handle.income_stmt
    except Exception as error:
        print(f"  {ticker}: statements failed ({error})")
        return None

    revenue = pick_row(income, "revenue")
    if revenue is None or revenue.empty:
        return None

    columns = {
        "revenue": revenue,
        "net_income": pick_row(income, "net_income"),
        "eps": pick_row(income, "eps"),
    }

    frame = pd.DataFrame(
        {name: series for name, series in columns.items() if series is not None}
    )

    # The statements do not always share period ends, and building a frame from
    # their Series unions the indices -- leaving rows with no revenue that would
    # break the growth shift.
    frame = frame.dropna(subset=["revenue"])
    if frame.empty:
        return None

    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    frame["ticker"] = ticker
    frame.index.name = "period_end"

    return frame.reset_index()


def ticker_earnings(ticker_handle, ticker):
    """Every earnings date yfinance will serve, past and scheduled."""
    try:
        dates = ticker_handle.get_earnings_dates(limit=60)
    except Exception as error:
        print(f"  {ticker}: earnings dates failed ({error})")
        return None

    if dates is None or dates.empty:
        return None

    stamps = pd.to_datetime(dates.index)
    if stamps.tz is not None:
        stamps = stamps.tz_localize(None)

    return pd.DataFrame({
        "ticker": ticker,
        "earnings_date": stamps.normalize(),
    })


def download_fundamentals(tickers,
                          statements_path=RAW_PATH / "fundamentals.parquet",
                          earnings_path=RAW_PATH / "earnings_dates.parquet"):
    """Fetch annual statements and earnings dates, and cache both."""
    statement_frames = []
    earnings_frames = []

    for n, ticker in enumerate(tickers, 1):
        handle = yf.Ticker(ticker)

        statements = ticker_statements(handle, ticker)
        if statements is not None:
            statement_frames.append(statements)

        earnings = ticker_earnings(handle, ticker)
        if earnings is not None:
            earnings_frames.append(earnings)

        if n % 10 == 0:
            print(f"  {n}/{len(tickers)} tickers")
        time.sleep(0.2)

    if not statement_frames:
        raise RuntimeError("No statements downloaded for any ticker.")

    statements = pd.concat(statement_frames, ignore_index=True)
    statements.to_parquet(statements_path, index=False)
    print(f"\nStatements: {len(statements)} ticker-years, "
          f"{statements['ticker'].nunique()} tickers, "
          f"{statements['period_end'].min().date()} to {statements['period_end'].max().date()}")

    if earnings_frames:
        earnings = pd.concat(earnings_frames, ignore_index=True)
        earnings.to_parquet(earnings_path, index=False)
        print(f"Earnings dates: {len(earnings)} reports, "
              f"{earnings['ticker'].nunique()} tickers, "
              f"{earnings['earnings_date'].min().date()} to {earnings['earnings_date'].max().date()}")

    return statements


def build_fundamental_features(raw):
    """Annual figures turned into ratios, dated from when they were public."""
    raw = raw.sort_values(["ticker", "period_end"]).copy()
    grouped = raw.groupby("ticker")

    # Annual, so the prior year is one row back rather than four.
    raw["revenue_growth_yoy"] = grouped["revenue"].transform(
        lambda s: s / s.shift(1).replace(0, np.nan) - 1
    )

    if "net_income" in raw.columns:
        raw["profit_margin"] = raw["net_income"] / raw["revenue"].replace(0, np.nan)

    raw["available_from"] = raw["period_end"] + pd.Timedelta(days=REPORTING_LAG_DAYS)

    keep = ["ticker", "available_from", "revenue_growth_yoy", "profit_margin", "eps"]
    keep = [column for column in keep if column in raw.columns]

    return raw[keep].dropna(subset=["available_from"]).sort_values("available_from")


def align_keys(left, right, left_on, right_on):
    """merge_asof rejects join keys that differ in dtype or datetime resolution."""
    left = left.copy()
    right = right.copy()

    left["ticker"] = left["ticker"].astype("string")
    right["ticker"] = right["ticker"].astype("string")

    left[left_on] = left[left_on].astype("datetime64[ns]")
    right[right_on] = right[right_on].astype("datetime64[ns]")

    return left.sort_values(left_on).reset_index(drop=True), \
        right.sort_values(right_on).reset_index(drop=True)


def merge_fundamentals(df, features):
    """Attach each daily row the most recent figures that were public by then."""
    df, features = align_keys(df, features, "date", "available_from")

    merged = pd.merge_asof(
        df, features,
        left_on="date", right_on="available_from",
        by="ticker", direction="backward",
    )

    # Price moves daily while the earnings figure holds, so this is built after
    # the join rather than in build_fundamental_features.
    if "eps" in merged.columns:
        merged["pe_ratio"] = merged["close"] / merged["eps"].replace(0, np.nan)

    return merged.drop(columns=["available_from", "eps"], errors="ignore")


def merge_earnings(df, earnings):
    """Days since the last report and days until the next, per row."""
    df, earnings = align_keys(df, earnings, "date", "earnings_date")

    previous = pd.merge_asof(
        df, earnings.rename(columns={"earnings_date": "prev_earnings"}),
        left_on="date", right_on="prev_earnings",
        by="ticker", direction="backward",
    )

    upcoming = pd.merge_asof(
        df, earnings.rename(columns={"earnings_date": "next_earnings"}),
        left_on="date", right_on="next_earnings",
        by="ticker", direction="forward",
    )

    # merge_asof preserves the left frame's row order and length, so these line
    # up positionally with df.
    df["days_since_earnings"] = (
        df["date"] - previous["prev_earnings"].to_numpy()
    ).dt.days
    df["days_to_earnings"] = (
        upcoming["next_earnings"].to_numpy() - df["date"]
    ).dt.days

    df["near_earnings"] = (
        (df["days_to_earnings"] <= EARNINGS_EXCLUSION_DAYS)
        | (df["days_since_earnings"] <= EARNINGS_EXCLUSION_DAYS)
    ).fillna(False)

    return df


def attach_fundamentals(df, earnings_path=RAW_PATH / "earnings_dates.parquet"):
    """Merge the earnings calendar and report coverage.

    Statement figures are no longer merged here -- edgar.attach_edgar supplies
    pe_ratio, profit_margin and revenue_growth_yoy with real filing dates and
    ~15 years of history. Merging both would collide on those column names.
    build_fundamental_features and merge_fundamentals are kept as the yfinance
    fallback if EDGAR is ever unavailable.
    """
    if earnings_path.exists():
        df = merge_earnings(df, pd.read_parquet(earnings_path))
    else:
        print(f"No {earnings_path} -- run fundamentals.py to fetch them")
        return df

    present = [column for column in FUNDAMENTAL_FEATURES if column in df.columns]
    if present:
        print("\nEarnings coverage (share of rows with a value):")
        print(df[present].notna().mean().to_string(float_format=lambda v: f"{v:.3f}"))

    if "near_earnings" in df.columns:
        print(f"Rows inside the {EARNINGS_EXCLUSION_DAYS}-day earnings window: "
              f"{df['near_earnings'].mean():.3f}")

    return df


if __name__ == "__main__":
    from extract_data import TICKERS

    download_fundamentals(TICKERS)
