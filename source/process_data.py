from pathlib import Path
import pandas as pd
import numpy as np

from fundamentals import attach_fundamentals
from edgar_fundamentals import attach_edgar

RAW_PATH = Path("data/raw_data")
PROCESSED_PATH = Path("data/processed_data")

# Forward-return horizons written into the processed panel: the ticker targets
# (future_return_Nd, beats_median_Nd) and the SPY benchmark (spy_future_return_Nd).
# Every horizon walkforward.HORIZON might be set to has to be listed here.
HORIZONS = [1, 5, 20]

NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
]


def clean_data(df):
    # Convert ticker -> OHLCV MultiIndex columns into rows
    processed_stocks = []

    tickers = df.columns.get_level_values(0).unique()

    for ticker in tickers:
        stock_df = df[ticker].copy()

        # Convert the date index into a normal column
        stock_df = stock_df.reset_index()

        # Add the ticker as its own column
        stock_df["ticker"] = ticker

        # Normalize column names
        stock_df.columns = [
            str(column).lower().replace(" ", "_")
            for column in stock_df.columns
        ]

        processed_stocks.append(stock_df)

    # Combine every stock into one dataframe
    df = pd.concat(
        processed_stocks,
        ignore_index=True
    )

    # Rename common date column names if necessary
    if "datetime" in df.columns:
        df = df.rename(columns={"datetime": "date"})

    if "index" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"index": "date"})

    required_columns = [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    # Make sure all required columns exist
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Keep only the columns needed for the MVP
    df = df[required_columns].copy()

    # Convert date into datetime
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    # Clean ticker values
    df["ticker"] = (
        df["ticker"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    # Treat empty ticker strings as missing values
    df.loc[
        df["ticker"] == "",
        "ticker"
    ] = pd.NA

    # Convert OHLCV columns into numeric values
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    original_rows = len(df)

    # Remove rows containing at least one missing value
    missing_rows = df.isna().any(axis=1)
    number_missing = missing_rows.sum()

    df = df.loc[~missing_rows].copy()

    # Remove duplicate ticker/date entries
    duplicate_rows = df.duplicated(
        subset=["ticker", "date"],
        keep="first"
    )

    number_duplicates = duplicate_rows.sum()

    df = df.loc[~duplicate_rows].copy()

    # Remove rows containing any negative OHLCV value
    negative_rows = (
        df[NUMERIC_COLUMNS] < 0
    ).any(axis=1)

    number_negative = negative_rows.sum()

    df = df.loc[~negative_rows].copy()

    # Remove rows containing impossible OHLC relationships
    invalid_ohlc = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    )

    number_invalid_ohlc = invalid_ohlc.sum()

    df = df.loc[~invalid_ohlc].copy()

    # Sort every stock chronologically
    df = df.sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)

    # Final validation
    assert not df.isna().any().any()

    assert not df.duplicated(
        subset=["ticker", "date"]
    ).any()

    assert not (
        df[NUMERIC_COLUMNS] < 0
    ).any().any()

    assert (df["high"] >= df["low"]).all()
    assert (df["high"] >= df["open"]).all()
    assert (df["high"] >= df["close"]).all()
    assert (df["low"] <= df["open"]).all()
    assert (df["low"] <= df["close"]).all()

    # Print processing results
    print("\nDATA PROCESSING COMPLETE")

    print(f"Rows before cleaning: {original_rows}")
    print(f"Empty rows removed: {number_missing}")
    print(f"Duplicate rows removed: {number_duplicates}")
    print(f"Negative rows removed: {number_negative}")
    print(f"Invalid OHLC rows removed: {number_invalid_ohlc}")
    print(f"Final rows: {len(df)}")


    print(df.head())
    print("Number of columns:", df.shape[1])
    print("Columns:", df.columns.tolist())

    return df

# used for liqudity and structure calculations
def add_prior_period_extremes(df, freq, label):
    df = df.assign(_period=df["date"].dt.to_period(freq))

    extremes = (
        df.groupby(["ticker", "_period"])
        .agg(_high=("high", "max"), _low=("low", "min"))
        .reset_index()
    )

    extremes[f"{label}_high"] = extremes.groupby("ticker")["_high"].shift(1)
    extremes[f"{label}_low"] = extremes.groupby("ticker")["_low"].shift(1)

    df = df.merge(
        extremes[["ticker", "_period", f"{label}_high", f"{label}_low"]],
        on=["ticker", "_period"],
        how="left",
    )

    df[f"close_vs_{label}_high"] = df["close"] / df[f"{label}_high"] - 1
    df[f"close_vs_{label}_low"] = df["close"] / df[f"{label}_low"] - 1

    return df.drop(columns=["_period", f"{label}_high", f"{label}_low"])


def extract_market_features(raw):
    """Index-level context: one row per date, identical across tickers.

    These cannot separate one stock from another on a given day, so they add
    nothing to a purely cross-sectional split. What they do is let the model
    condition on the regime -- the same RSI reading means something different in
    a calm tape than in a panic.
    """
    spy = pd.to_numeric(raw["SPY"]["Close"], errors="coerce")
    vix = pd.to_numeric(raw["^VIX"]["Close"], errors="coerce")

    market = pd.DataFrame({
        "spy_return_1d": spy.pct_change(),
        "spy_return_5d": spy.pct_change(periods=5),
        "spy_vs_ema50": spy / spy.ewm(span=50, adjust=False).mean() - 1,
        "vix_level": vix,
        "vix_vs_60d": vix / vix.rolling(window=60).mean() - 1,
    })

    # Forward SPY returns -- a benchmark, never a feature. The equal-weight
    # universe average is a hindsight portfolio: those 90 tickers are the 2026
    # index members held back to 2023, so it contains only companies that stayed
    # in. SPY is what someone could actually have bought.
    #
    # These look forward and must stay out of MARKET_FEATURES.
    for horizon in HORIZONS:
        market[f"spy_future_return_{horizon}d"] = spy.shift(-horizon) / spy - 1

    market = market.reset_index()
    market = market.rename(columns={market.columns[0]: "date"})
    market["date"] = pd.to_datetime(market["date"]).astype("datetime64[ns]")

    return market


def extract_technicals(df) :
    df = df.sort_values(["ticker", "date"]).copy()

    # Calculate exponential moving averages
    df["ema_20"] = df.groupby("ticker")["close"].transform(lambda x: x.ewm(span=20, adjust=False).mean())
    df["ema_50"] = df.groupby("ticker")["close"].transform(lambda x: x.ewm(span=50, adjust=False).mean())
    df["ema_100"] = df.groupby("ticker")["close"].transform(lambda x: x.ewm(span=100, adjust=False).mean())
    
    # Calculate close price relative to EMA
    df["close_vs_ema20"] = df["close"] / df["ema_20"] - 1
    df["close_vs_ema50"] = df["close"] / df["ema_50"] - 1
    df["close_vs_ema100"] = df["close"] / df["ema_100"] - 1
    
    # Returns
    df["return_5d"] = df.groupby("ticker")["close"].pct_change(periods=5)
    df["return_10d"] = df.groupby("ticker")["close"].pct_change(periods=10)
    df["return_20d"] = df.groupby("ticker")["close"].pct_change(periods=20)

    # Daily return
    df["daily_return"] = (df.groupby("ticker")["close"].pct_change())

    # Volatility
    df["volatility_5d"] = df.groupby("ticker")["daily_return"].transform(lambda x: x.rolling(window=5).std())
    df["volatility_20d"] = df.groupby("ticker")["daily_return"].transform(lambda x: x.rolling(window=20).std())
    
    # RSI 14 day
    delta = df.groupby("ticker")["close"].transform(lambda x: x.diff())
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.groupby(df["ticker"]).transform(lambda x: x.rolling(window=14).mean())
    avg_loss = loss.groupby(df["ticker"]).transform(lambda x: x.rolling(window=14).mean())
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # Volume, relative to the ticker's own recent norm so it is comparable
    # across names. Raw share counts differ by orders of magnitude.
    df["volume_vs_20d"] = df["volume"] / df.groupby("ticker")["volume"].transform(
        lambda x: x.rolling(window=20).mean()
    ).replace(0, np.nan)
    df["volume_vs_60d"] = df["volume"] / df.groupby("ticker")["volume"].transform(
        lambda x: x.rolling(window=60).mean()
    ).replace(0, np.nan)

    # Reversals
    sma_20 = df.groupby("ticker")["close"].transform(lambda x: x.rolling(window=20).mean())
    std_20 = df.groupby("ticker")["close"].transform(lambda x: x.rolling(window=20).std())
    df["bollinger_zscore"] = (df["close"] - sma_20) / std_20.replace(0, np.nan)

    high_20d = df.groupby("ticker")["high"].transform(lambda x: x.rolling(window=20).max())
    low_20d = df.groupby("ticker")["low"].transform(lambda x: x.rolling(window=20).min())

    # 0 = sitting on the 20d low, 1 = sitting on the 20d high
    df["range_position_20d"] = (df["close"] - low_20d) / (high_20d - low_20d).replace(0, np.nan)

    # Distance from recent extremes
    df["close_vs_20d_high"] = df["close"] / high_20d - 1
    df["close_vs_20d_low"] = df["close"] / low_20d - 1

    high_50d = df.groupby("ticker")["high"].transform(lambda x: x.rolling(window=50).max())
    low_50d = df.groupby("ticker")["low"].transform(lambda x: x.rolling(window=50).min())
    df["close_vs_50d_high"] = df["close"] / high_50d - 1
    df["close_vs_50d_low"] = df["close"] / low_50d - 1

    # Liquidity + structure: previous completed week and month
    df = add_prior_period_extremes(df, "W", "prev_week")
    df = add_prior_period_extremes(df, "M", "prev_month")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Rejection
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    body_top = df[["open", "close"]].max(axis=1)
    body_bottom = df[["open", "close"]].min(axis=1)

    df["upper_wick_pct"] = (df["high"] - body_top) / candle_range
    df["lower_wick_pct"] = (body_bottom - df["low"]) / candle_range

    # Body size relative to candle high and low
    df["body_pct"] = (df["close"] - df["open"]) / candle_range

    # y predictors (forward returns). Every horizon is kept so TARGET can be
    # switched without reprocessing.
    #
    # The cost: dropna() below removes any row missing a target, so the longest
    # horizon decides how much of the panel tail is lost -- 20 trading days per
    # ticker rather than 5.
    for horizon in HORIZONS:
        df[f"future_return_{horizon}d"] = (
            df.groupby("ticker")["close"].shift(-horizon) / df["close"] - 1
        )

    return df

def process_data(RAW_DATA_PATH, PROCESSED_DATA_PATH):
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {RAW_DATA_PATH}"
        )

    df = pd.read_parquet(RAW_DATA_PATH)
    print("Raw data loaded successfully")
    df = clean_data(df)
    print("Data cleaned successfully")
    df = extract_technicals(df)
    print("Technical indicators extracted successfully")
    df = df.dropna().reset_index(drop=True)
    print("Null technical rows removed")

    # Beating the MEDIAN: did this ticker land in the better half of the day's returns.
    for horizon in HORIZONS:
        df[f"beats_median_{horizon}d"] = (
            df.groupby("date")[f"future_return_{horizon}d"].rank(pct=True) > 0.5
        ).astype(int)

    # Merged after dropna so sparse extras cannot delete technical rows.
    market_path = RAW_PATH / "market.parquet"
    if market_path.exists():
        market = extract_market_features(pd.read_parquet(market_path))
        df["date"] = df["date"].astype("datetime64[ns]")
        df = df.merge(market, on="date", how="left")
        print(f"Market features merged "
              f"({df['vix_level'].notna().mean():.3f} coverage)")
    else:
        print(f"No {market_path} -- run extract_data.py to fetch it")

    # Earnings dates come from yfinance; the statement figures come from EDGAR,
    # which carries real filing dates and ~15 years instead of 4.
    df = attach_fundamentals(df)
    df = attach_edgar(df)
    print("Final data shape:", df.shape)
    print("Final data columns:", df.columns.tolist())
    # Save processed data
    df.to_parquet(PROCESSED_DATA_PATH, index=False)
    
    return df

if __name__ == "__main__":
    
    #process_data(
    #    RAW_DATA_PATH=RAW_PATH / "output.parquet",
    #    PROCESSED_DATA_PATH=PROCESSED_PATH / "cleaned_output.parquet"
    #)

    #process_data(
    #    RAW_DATA_PATH=RAW_PATH / "test_output.parquet",
    #    PROCESSED_DATA_PATH=PROCESSED_PATH / "cleaned_test_output.parquet"
    #)
    
    df = process_data(
        RAW_DATA_PATH=RAW_PATH / "output.parquet",
        PROCESSED_DATA_PATH=PROCESSED_PATH / "cleaned_output.parquet"
    )
    
    split_date = "2025-01-01"
    
    train_data = df[df["date"] < split_date]
    test_data = df[df["date"] >= split_date]
    
    train_data.to_parquet(PROCESSED_PATH / "cleaned_train_output.parquet", index=False)
    test_data.to_parquet(PROCESSED_PATH / "cleaned_test_output.parquet", index=False)