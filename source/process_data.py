from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw_data")
PROCESSED_PATH = Path("data/processed_data")

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

def extract_technicals(df) :
    # Calculate technical indicators for each stock
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
    
    # Calculate relative strength index (RSI)
    delta = df.groupby("ticker")["close"].transform(lambda x: x.diff())
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.groupby(df["ticker"]).transform(lambda x: x.rolling(window=14).mean())
    avg_loss = loss.groupby(df["ticker"]).transform(lambda x: x.rolling(window=14).mean())
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    
    # y predictors (outputs)
    df["future_return_5d"] = (df.groupby("ticker")["close"].shift(-5) / df["close"] - 1)

    return df

def process_data(RAW_DATA_PATH, PROCESSED_DATA_PATH):
    # Check that the raw data file exists
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {RAW_DATA_PATH}"
        )

    # Load raw data
    df = pd.read_parquet(RAW_DATA_PATH)
    print("Raw data loaded successfully")
    df = clean_data(df)
    print("Data cleaned successfully")
    df = extract_technicals(df)
    print("Technical indicators extracted successfully")
    df = df.dropna().reset_index(drop=True)
    print("Null technical rows removed")
    print("Final data shape:", df.shape)
    print("Final data columns:", df.columns.tolist())
    # Save processed data
    df.to_parquet(PROCESSED_DATA_PATH, index=False)

if __name__ == "__main__":
    process_data(
        RAW_DATA_PATH=RAW_PATH / "output.parquet",
        PROCESSED_DATA_PATH=PROCESSED_PATH / "cleaned_output.parquet"
    )
    
    process_data(
        RAW_DATA_PATH=RAW_PATH / "test_output.parquet",
        PROCESSED_DATA_PATH=PROCESSED_PATH / "cleaned_test_output.parquet"
    )