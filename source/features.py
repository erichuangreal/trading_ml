import pandas as pd


BASE_FEATURES = [
    # Trend
    "close_vs_ema20",
    "close_vs_ema50",
    "close_vs_ema100",

    # Momentum
    "return_5d",
    "return_10d",
    "return_20d",

    # Volatility
    "volatility_5d",
    "volatility_20d",

    # Reversal / exhaustion
    "rsi",
    "bollinger_zscore",
    "range_position_20d",

    # Liquidity / structure
    "close_vs_prev_week_high",
    "close_vs_prev_week_low",
    "close_vs_prev_month_high",
    "close_vs_prev_month_low",
    "close_vs_20d_high",
    "close_vs_20d_low",
    "close_vs_50d_high",
    "close_vs_50d_low",

    # Rejection
    "upper_wick_pct",
    "lower_wick_pct",
    "body_pct",
]

# Quarterly figures, joined on filing date rather than period end. Sparse before
# ~2024 because yfinance only serves a few quarters back, and left NaN rather
# than filled -- XGBoost learns a direction for missing, RandomForest cannot.
FUNDAMENTAL_FEATURES = [
    "revenue_growth_yoy",
    "eps_growth_yoy",
    "profit_margin",
    "debt_to_equity",
    "fcf_margin",
    "pe_ratio",
    "ps_ratio",
]

# Set False when data/raw_data/fundamentals.parquet has not been fetched, or the
# columns will be listed here but missing from the parquet and every lookup fails.
USE_FUNDAMENTALS = False

FEATURES = (
    BASE_FEATURES
    + [f"{column}_rank" for column in BASE_FEATURES]
    + (FUNDAMENTAL_FEATURES if USE_FUNDAMENTALS else [])
)


def add_cross_sectional_ranks(df, features=BASE_FEATURES):
    # Percentile rank of each feature within each day, across all tickers.
    ranks = {
        f"{column}_rank": df.groupby("date")[column].rank(pct=True)
        for column in features
    }

    return df.assign(**ranks)
