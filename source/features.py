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

# Index level, identical across tickers on a date. They cannot separate one
# stock from another, so they add nothing to a purely cross-sectional split --
# what they offer is regime: the same RSI reading means something different in a
# calm tape than in a panic.
MARKET_FEATURES = [
    "spy_return_1d",
    "spy_return_5d",
    "spy_vs_ema50",
    "vix_level",
    "vix_vs_60d",
]



# From the yfinance earnings calendar. These change every row and cover the whole
# panel, unlike the statement figures.
FUNDAMENTAL_FEATURES = [
    "days_since_earnings",
    "days_to_earnings",
]

# From SEC EDGAR XBRL, dated by the day each filing actually landed rather than
# by a lag estimate. Quarterly back to ~2009, so unlike the yfinance statements
# these exist across the whole panel and TTM windows have enough history.
EDGAR_FEATURES = [
    "pe_ratio",
    "profit_margin",
    "revenue_growth_yoy",
]

# All off. Four separate additions to the 22-technical base have now been tested
# against it and every one lowered the edge:
#
#   +22 cross-sectional ranks   -0.35pp
#   +2  volatility ranks        -0.33pp
#   +8  market and EDGAR        -0.35pp
#   +2  earnings distance       -0.48pp
#
# No single comparison clears the +-0.51pp noise on a difference of two runs, but
# four consecutive negatives would happen by chance about 6% of the time. The
# working read is that at ~0.5pp of signal across 130k rows, an extra column
# gives the trees another way to fit noise and that costs more than it adds.
# Feature usefulness turns out to be horizon-dependent, so these are on at 5-day
# and should go off at 1-day:
#
#   1-day target: four separate additions each cost ~0.35pp
#   5-day target: removing the same features cost 1.09pp (+1.23pp -> +0.14pp)
#
# The likely reason is timescale. VIX level, market trend, P/E and margins move
# over weeks or quarters -- too slow to say anything about tomorrow, but matched
# to a week. Against a 1-day target they were noise the trees could overfit.
USE_MARKET = True
USE_EDGAR = True
USE_FUNDAMENTALS = True

FEATURES = (
    BASE_FEATURES
    + (MARKET_FEATURES if USE_MARKET else [])
    + (FUNDAMENTAL_FEATURES if USE_FUNDAMENTALS else [])
    + (EDGAR_FEATURES if USE_EDGAR else [])
)
