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

data = yf.download(" ".join(TICKERS), start="2020-01-01", end="2024-01-01", interval = "1d", group_by = 'tickers')
df = pd.DataFrame(data)

df.to_parquet('data/raw_data/output.parquet', engine='pyarrow', compression='snappy')

df = pd.read_parquet("data/raw_data/output.parquet")

print(df.head())
print(df.shape)
print(df["AAPL"]["Open"])