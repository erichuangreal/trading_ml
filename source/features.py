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
]

FUNDAMENTAL_FEATURES = [
    "revenue_growth_yoy",
    "eps_growth_yoy",
    "profit_margin",
    "debt_to_equity",
    "fcf_margin",
    "pe_ratio",
    "ps_ratio",
]

USE_FUNDAMENTALS = False

FEATURES = (
    BASE_FEATURES
    + (FUNDAMENTAL_FEATURES if USE_FUNDAMENTALS else [])
)
