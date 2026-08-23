import yfinance as yf
import pandas as pd

TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "WMT",
    "GOOGL",
    "META",
    "AMD",
    "JNJ",
    "MU",
    "JPM"
]

def download_data(start, end, output_path) :
    data = yf.download(" ".join(TICKERS), start = start, end = end, interval = "1d", group_by = 'tickers')
    df = pd.DataFrame(data)
    
    if df.empty:
        print("WARNING: No data was downloaded!")
        return df
    
    df.to_parquet(output_path, engine='pyarrow', compression='snappy')
    
    print(df.head())
    print(df.shape)
    
    return df

if __name__ == "__main__":
    training_data = download_data("2020-01-01", "2025-01-01", "data/raw_data/output.parquet")
    print("Training data downloaded and saved.")
    test_data = download_data("2025-01-02", "2025-02-02", "data/raw_data/test_output.parquet")
    print("Test data downloaded and saved.")
