from pathlib import Path
import pandas as pd


RAW_DATA_PATH = Path("data/raw_data/output.parquet")
PROCESSED_DATA_PATH = Path("data/processed_data/cleaned_output.parquet")

NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
]


def process_data():
    # Check that the raw data file exists
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {RAW_DATA_PATH}"
        )

    # Load raw data
    df = pd.read_parquet(RAW_DATA_PATH)

    print("Original data shape:", df.shape)
    print("Original columns:")
    print(df.columns)

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

    # Save the cleaned dataset
    df.to_parquet(
        PROCESSED_DATA_PATH,
        index=False
    )

    # Print processing results
    print("\nDATA PROCESSING COMPLETE")

    print(f"Rows before cleaning: {original_rows}")
    print(f"Empty rows removed: {number_missing}")
    print(f"Duplicate rows removed: {number_duplicates}")
    print(f"Negative rows removed: {number_negative}")
    print(f"Invalid OHLC rows removed: {number_invalid_ohlc}")
    print(f"Final rows: {len(df)}")

    print(
        f"Unique tickers: {df['ticker'].nunique()}"
    )

    print(
        f"Date range: "
        f"{df['date'].min()} -> {df['date'].max()}"
    )

    print(
        f"Saved to: {PROCESSED_DATA_PATH}"
    )

    print("\nRows per ticker:")
    print(
        df.groupby("ticker")
        .size()
        .sort_index()
    )

    print("\nProcessed data preview:")
    print(df.head())

    return df


if __name__ == "__main__":
    process_data()