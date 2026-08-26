"""Fundamental features from yfinance quarterly statements.

Two things make this harder than the technicals:

1. Coverage. quarterly_income_stmt returns roughly 4-6 quarters, so most of a
   2020-2025 panel has no fundamental data at all. Those rows stay NaN. XGBoost
   handles NaN natively; RandomForest does not, so this is XGBoost-only.

2. Lookahead. The column dates on these statements are fiscal period ENDS, not
   filing dates. A quarter ending 2025-03-31 is not public until early May, so
   joining on the period end would hand the model numbers 45+ days before anyone
   could have seen them. REPORTING_LAG_DAYS shifts each figure forward to a date
   it was plausibly known, and merge_asof then gives every daily row the most
   recent figure that was public by then.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
import time

RAW_PATH = Path("data/raw_data")

# Calendar days between fiscal period end and the figure being public. US large
# caps file 10-Qs within 40 days; 45 is a slightly conservative round number.
REPORTING_LAG_DAYS = 45

FUNDAMENTAL_FEATURES = [
    "revenue_growth_yoy",
    "eps_growth_yoy",
    "profit_margin",
    "debt_to_equity",
    "fcf_margin",
    "pe_ratio",
    "ps_ratio",
]

# yfinance row labels vary by ticker and change between versions, so each figure
# is looked up against a list of candidates rather than one exact string.
ROW_CANDIDATES = {
    "revenue": ["Total Revenue", "Operating Revenue"],
    "net_income": [
        "Net Income",
        "Net Income Common Stockholders",
        "Net Income From Continuing Operation Net Minority Interest",
    ],
    "eps": ["Diluted EPS", "Basic EPS"],
    "shares": ["Diluted Average Shares", "Basic Average Shares"],
    "total_debt": ["Total Debt"],
    "long_term_debt": ["Long Term Debt"],
    "current_debt": ["Current Debt", "Current Debt And Capital Lease Obligation"],
    "equity": ["Stockholders Equity", "Total Equity Gross Minority Interest"],
    "shares_balance": ["Ordinary Shares Number", "Share Issued"],
    "free_cash_flow": ["Free Cash Flow"],
    "operating_cash_flow": ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"],
    "capex": ["Capital Expenditure"],
}


def pick_row(statement, key):
    """First matching row from a yfinance statement, or None."""
    if statement is None or statement.empty:
        return None

    for name in ROW_CANDIDATES[key]:
        if name in statement.index:
            return pd.to_numeric(statement.loc[name], errors="coerce")

    return None


def ticker_fundamentals(ticker):
    """Quarterly figures for one ticker, oldest first, indexed by period end."""
    handle = yf.Ticker(ticker)

    try:
        income = handle.quarterly_income_stmt
        balance = handle.quarterly_balance_sheet
        cashflow = handle.quarterly_cashflow
    except Exception as error:
        print(f"  {ticker}: fetch failed ({error})")
        return None

    revenue = pick_row(income, "revenue")
    if revenue is None or revenue.empty:
        print(f"  {ticker}: no revenue row, skipping")
        return None

    columns = {
        "revenue": revenue,
        "net_income": pick_row(income, "net_income"),
        "eps": pick_row(income, "eps"),
        "shares": pick_row(income, "shares"),
        "total_debt": pick_row(balance, "total_debt"),
        "equity": pick_row(balance, "equity"),
        "free_cash_flow": pick_row(cashflow, "free_cash_flow"),
    }

    # Total Debt is often absent; rebuild it from the two components.
    if columns["total_debt"] is None:
        long_term = pick_row(balance, "long_term_debt")
        current = pick_row(balance, "current_debt")
        if long_term is not None:
            columns["total_debt"] = long_term.add(
                current if current is not None else 0, fill_value=0
            )

    # Same for Free Cash Flow: operating cash flow less capital expenditure.
    if columns["free_cash_flow"] is None:
        operating = pick_row(cashflow, "operating_cash_flow")
        capex = pick_row(cashflow, "capex")
        if operating is not None and capex is not None:
            # capex is reported negative, so adding it subtracts the spend.
            columns["free_cash_flow"] = operating.add(capex, fill_value=0)

    # Share count can come from either statement.
    if columns["shares"] is None:
        columns["shares"] = pick_row(balance, "shares_balance")

    frame = pd.DataFrame(
        {name: series for name, series in columns.items() if series is not None}
    )

    if frame.empty:
        return None

    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()

    # The three statements do not always report the same period ends, and building
    # a DataFrame from their Series unions the indices. That leaves phantom rows
    # with no income-statement figures, which then sit inside the rolling windows
    # and turn every TTM touching them into NaN.
    frame = frame.dropna(subset=["revenue"])

    frame["ticker"] = ticker
    frame.index.name = "period_end"

    return frame.reset_index()


def download_fundamentals(tickers, output_path=RAW_PATH / "fundamentals.parquet"):
    """Fetch every ticker's quarterly statements and cache them.

    Three API calls per ticker, so this is slow and worth doing once rather than
    on every processing run.
    """
    frames = []

    for n, ticker in enumerate(tickers, 1):
        frame = ticker_fundamentals(ticker)
        if frame is not None:
            frames.append(frame)

        if n % 10 == 0:
            print(f"  {n}/{len(tickers)} tickers")
        time.sleep(0.2)

    if not frames:
        raise RuntimeError("No fundamentals downloaded for any ticker.")

    raw = pd.concat(frames, ignore_index=True)
    raw.to_parquet(output_path, index=False)

    print(f"\nFundamentals: {len(raw)} ticker-quarters, "
          f"{raw['ticker'].nunique()} tickers, "
          f"{raw['period_end'].min().date()} to {raw['period_end'].max().date()}")

    return raw


def build_fundamental_features(raw):
    """Turn raw quarterly figures into ratios, dated from when they were public."""
    raw = raw.sort_values(["ticker", "period_end"]).copy()
    grouped = raw.groupby("ticker")

    # Flow items are summed over four quarters; balance items are point-in-time.
    for column in ["revenue", "net_income", "eps", "free_cash_flow"]:
        if column in raw.columns:
            raw[f"ttm_{column}"] = grouped[column].transform(
                lambda s: s.rolling(4, min_periods=4).sum()
            )

    # Same quarter a year earlier is four rows back.
    if "revenue" in raw.columns:
        raw["revenue_growth_yoy"] = grouped["revenue"].transform(
            lambda s: s / s.shift(4).replace(0, np.nan) - 1
        )

    if "eps" in raw.columns:
        raw["eps_growth_yoy"] = grouped["eps"].transform(
            lambda s: s / s.shift(4).replace(0, np.nan) - 1
        )

    if "ttm_net_income" in raw.columns and "ttm_revenue" in raw.columns:
        raw["profit_margin"] = raw["ttm_net_income"] / raw["ttm_revenue"].replace(0, np.nan)

    if "total_debt" in raw.columns and "equity" in raw.columns:
        raw["debt_to_equity"] = raw["total_debt"] / raw["equity"].replace(0, np.nan)

    if "ttm_free_cash_flow" in raw.columns and "ttm_revenue" in raw.columns:
        raw["fcf_margin"] = raw["ttm_free_cash_flow"] / raw["ttm_revenue"].replace(0, np.nan)

    # The date a daily row is first allowed to see these numbers.
    raw["available_from"] = raw["period_end"] + pd.Timedelta(days=REPORTING_LAG_DAYS)

    keep = [
        "ticker",
        "available_from",
        "revenue_growth_yoy",
        "eps_growth_yoy",
        "profit_margin",
        "debt_to_equity",
        "fcf_margin",
        "ttm_eps",
        "ttm_revenue",
        "shares",
    ]

    keep = [column for column in keep if column in raw.columns]

    return raw[keep].dropna(subset=["available_from"]).sort_values("available_from")


def merge_fundamentals(df, features):
    """Attach each daily row the most recent figures that were public by then.

    merge_asof with direction="backward" is the point-in-time join: a row dated
    2025-06-10 picks up the last filing available on or before that date, and
    carries it until the next one lands.
    """
    df = df.sort_values("date").reset_index(drop=True)
    features = features.sort_values("available_from").reset_index(drop=True)

    # merge_asof demands both join keys match exactly. Two mismatches to fix:
    # ticker is StringDtype on both sides but with different NA sentinels
    # (clean_data uses .astype("string"), these arrive as plain Python str), and
    # the dates land at different resolutions -- ms off the parquet, us from the
    # yfinance timestamps.
    df["ticker"] = df["ticker"].astype("string")
    features["ticker"] = features["ticker"].astype("string")

    df["date"] = df["date"].astype("datetime64[ns]")
    features["available_from"] = features["available_from"].astype("datetime64[ns]")

    merged = pd.merge_asof(
        df,
        features,
        left_on="date",
        right_on="available_from",
        by="ticker",
        direction="backward",
    )

    # P/E and P/S move daily because price does, so they are built after the join.
    if "ttm_eps" in merged.columns:
        merged["pe_ratio"] = merged["close"] / merged["ttm_eps"].replace(0, np.nan)

    if "ttm_revenue" in merged.columns and "shares" in merged.columns:
        market_cap = merged["close"] * merged["shares"]
        merged["ps_ratio"] = market_cap / merged["ttm_revenue"].replace(0, np.nan)

    coverage = merged[FUNDAMENTAL_FEATURES].notna().mean()
    print("\nFundamental coverage (share of rows with a value):")
    print(coverage.to_string(float_format=lambda v: f"{v:.3f}"))

    return merged.drop(columns=["available_from", "ttm_eps", "ttm_revenue", "shares"],
                       errors="ignore")


if __name__ == "__main__":
    from extract_data import TICKERS

    download_fundamentals(TICKERS)
