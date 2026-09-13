BASE_FEATURES = [
    "close_vs_ema20",
    "close_vs_ema50",
    "close_vs_ema100",

    "return_5d",
    "return_10d",
    "return_20d",

    "volatility_5d",
    "volatility_20d",

    "rsi",
    "bollinger_zscore",
    "range_position_20d",

    "close_vs_prev_week_high",
    "close_vs_prev_week_low",
    "close_vs_prev_month_high",
    "close_vs_prev_month_low",
    "close_vs_20d_high",
    "close_vs_20d_low",
    "close_vs_50d_high",
    "close_vs_50d_low",

    "upper_wick_pct",
    "lower_wick_pct",
    "body_pct",

    "volume_vs_20d",
    "volume_vs_60d",
]

MARKET_FEATURES = [
    "spy_return_1d",
    "spy_return_5d",
    "spy_vs_ema50",
    "vix_level",
    "vix_vs_60d",
]



# From the yfinance earnings calendar.
FUNDAMENTAL_FEATURES = [
    "days_since_earnings",
    "days_to_earnings",
]

# From SEC EDGAR XBRL.
EDGAR_FEATURES = [
    "pe_ratio",
    "profit_margin",
    "revenue_growth_yoy",
]

USE_MARKET = True
USE_EDGAR = True
USE_FUNDAMENTALS = True

FEATURES = (
    BASE_FEATURES
    + (MARKET_FEATURES if USE_MARKET else [])
    + (FUNDAMENTAL_FEATURES if USE_FUNDAMENTALS else [])
    + (EDGAR_FEATURES if USE_EDGAR else [])
)
