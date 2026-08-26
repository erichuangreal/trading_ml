import yfinance as yf
import pandas as pd

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

if __name__ == "__main__":
    output_data = download_data("2020-01-01", "2026-01-01", "data/raw_data/output.parquet")
    print("Data downloaded and saved.")

