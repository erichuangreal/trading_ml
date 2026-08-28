import yfinance as yf
import pandas as pd
from datetime import date, timedelta

START = "2015-01-01"

# yfinance's `end` is exclusive -- it stops the day before the date given -- so
# tomorrow is what includes today's bar. Hardcoding a date here silently caps the
# download: predict.py then scores a stale "as of" day no matter how often the
# script is re-run, and the staleness looks like a yfinance problem rather than
# a date problem.
#
# Today's bar only exists once the market has closed. Run before the close and
# the newest row is either missing or a partial day whose "close" is just the
# price at that moment. Weekends and holidays produce no bar at all, so a
# Saturday run ending on Friday is correct, not stale.
END = (date.today() + timedelta(days=1)).isoformat()

# NASDAQ-100 constituents that traded for the whole 2020-01-01 onward window.
# Post-2020 additions (ABNB, PLTR, ARM, GEHC, CEG, APP, WBD) are left out so
# every ticker contributes a full series and the panel stays balanced.
TICKERS = [
    # Semiconductors
    "NVDA",
    "AMD",
    "AVGO",
    "QCOM",
    "TXN",
    "INTC",
    "MU",
    "ADI",
    "AMAT",
    "LRCX",
    "KLAC",
    "NXPI",
    "MCHP",
    "MRVL",
    "ON",
    "ASML",
    "SWKS",
    "TER",

    # Software and internet
    "AAPL",
    "MSFT",
    "GOOGL",
    "META",
    "AMZN",
    "NFLX",
    "ADBE",
    "CSCO",
    "INTU",
    "SNPS",
    "CDNS",
    "ADSK",
    "WDAY",
    "PANW",
    "FTNT",
    "CRWD",
    "ZS",
    "DDOG",
    "TEAM",
    "MDB",
    "PYPL",
    "CDW",

    # Consumer and retail
    "TSLA",
    "COST",
    "PEP",
    "SBUX",
    "MDLZ",
    "MNST",
    "KDP",
    "KHC",
    "LULU",
    "ROST",
    "ORLY",
    "DLTR",
    "BKNG",
    "MAR",
    "EA",
    "TTWO",

    # Healthcare
    "AMGN",
    "GILD",
    "VRTX",
    "REGN",
    "MRNA",
    "BIIB",
    "ILMN",
    "IDXX",
    "DXCM",
    "ALGN",
    "ISRG",

    # Industrials and transport
    "HON",
    "CSX",
    "PCAR",
    "ODFL",
    "CPRT",
    "FAST",
    "PAYX",
    "ADP",
    "CTAS",
    "VRSK",

    # Media and telecom
    "CMCSA",
    "TMUS",
    "CHTR",

    # Utilities and energy
    "EXC",
    "XEL",
    "AEP",
    "BKR",

    # International
    "JD",
    "PDD",
    "MELI",

    # Other
    "SMCI",
    "TTD",
    "MSTR",
    "AXON",
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

# Index level, joined to every ticker by date. SPY for the market's own move and
# as the benchmark the baskets are measured against, VIX for whether the tape is
# calm or panicking -- the only feature here that speaks to regime rather than
# direction.
MARKET_TICKERS = ["SPY", "^VIX"]


def download_market(start, end, output_path):
    data = yf.download(" ".join(MARKET_TICKERS), start=start, end=end,
                       interval="1d", group_by='tickers')
    df = pd.DataFrame(data)

    if df.empty:
        print("WARNING: No market data was downloaded!")
        return df

    df.to_parquet(output_path, engine='pyarrow', compression='snappy')
    print(f"Market data: {df.shape}")

    return df


if __name__ == "__main__":
    print(f"Downloading {START} to {END} (exclusive -- newest bar is today's)")

    output_data = download_data(START, END, "data/raw_data/output.parquet")
    print("Data downloaded and saved.")
    market_data = download_market(START, END, "data/raw_data/market.parquet")
    print("Market data downloaded and saved.")

