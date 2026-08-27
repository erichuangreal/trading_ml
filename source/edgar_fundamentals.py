"""Point-in-time fundamentals from the SEC EDGAR XBRL API.

yfinance serves 4 annual periods with no filing dates, which left pe_ratio and
profit_margin absent for half of training and dated by a 90-day guess. EDGAR
publishes every XBRL fact from every filing, each carrying a `filed` date -- the
day it actually became public. That removes the guess entirely and reaches back
to roughly 2009.

Two wrinkles it introduces:

1. Tag names vary. Revenue might be Revenues, SalesRevenueNet, or
   RevenueFromContractWithCustomerExcludingAssessedTax depending on the filer and
   the year, so each figure is looked up against a list of candidates.

2. The same period is reported many times -- in its original filing, in
   amendments, and as a comparative in later filings. Only the earliest filing
   tells you when the number was first knowable, so periods are deduplicated on
   min(filed).
"""

import pandas as pd
import numpy as np
import requests
import time
import json
from pathlib import Path

RAW_PATH = Path("data/raw_data")

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# The SEC blocks requests without a descriptive User-Agent naming a contact.
# Put your own email here -- they will rate-limit or ban the placeholder.
USER_AGENT = "trading_ml research huangheeh@gmail.com"

# SEC allows 10 requests/second. This stays comfortably under.
REQUEST_DELAY = 0.15

# A quarterly fact covers ~90 days. Filters out the annual and half-year
# durations that share the same tags.
QUARTER_DAYS = (80, 100)

# us-gaap tags, in preference order.
CONCEPTS = {
    "revenue": (
        [
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
            "SalesRevenueGoodsNet",
        ],
        "USD",
    ),
    "net_income": (
        [
            "NetIncomeLoss",
            "ProfitLoss",
            "NetIncomeLossAvailableToCommonStockholdersBasic",
        ],
        "USD",
    ),
    "eps": (
        [
            "EarningsPerShareDiluted",
            "EarningsPerShareBasicAndDiluted",
            "EarningsPerShareBasic",
        ],
        "USD/shares",
    ),
}


def session():
    handle = requests.Session()
    handle.headers.update({"User-Agent": USER_AGENT})

    return handle


# company_tickers.json lists current registrants only, so a company that has
# since deregistered -- taken private, acquired -- drops out even though its
# filings are still on EDGAR. Look the CIK up at sec.gov/cgi-bin/browse-edgar by
# company name and add it here.
MANUAL_CIK = {
    # "EA": "0000712515",
}


def load_cik_map(handle):
    """Ticker -> zero-padded 10-digit CIK, which is what the API expects."""
    response = handle.get(TICKER_MAP_URL, timeout=30)
    response.raise_for_status()

    mapping = {
        entry["ticker"].upper(): str(entry["cik_str"]).zfill(10)
        for entry in response.json().values()
    }
    mapping.update({
        ticker.upper(): cik.zfill(10) for ticker, cik in MANUAL_CIK.items()
    })

    return mapping


def concept_frame(facts, tags, unit, name):
    """One concept as period_end / filed / value, earliest filing per period."""
    for tag in tags:
        concept = facts.get(tag)
        if not concept:
            continue

        rows = concept.get("units", {}).get(unit)
        if not rows:
            continue

        frame = pd.DataFrame(rows)
        if "start" not in frame.columns or frame.empty:
            continue

        frame["start"] = pd.to_datetime(frame["start"])
        frame["end"] = pd.to_datetime(frame["end"])
        frame["filed"] = pd.to_datetime(frame["filed"])

        # Quarterly only -- the same tag also carries annual and half-year facts.
        duration = (frame["end"] - frame["start"]).dt.days
        frame = frame[duration.between(*QUARTER_DAYS)]
        if frame.empty:
            continue

        # A period reappears in amendments and as a comparative in later
        # filings. The first filing is when it became knowable.
        frame = (
            frame.sort_values("filed")
            .groupby("end", as_index=False)
            .first()
        )

        return frame[["end", "filed", "val"]].rename(
            columns={"val": name, "filed": f"filed_{name}"}
        )

    return None


def ticker_facts(handle, ticker, cik):
    """Quarterly revenue, net income and EPS for one company."""
    try:
        response = handle.get(COMPANY_FACTS_URL.format(cik=cik), timeout=60)
        response.raise_for_status()
        facts = response.json()["facts"].get("us-gaap", {})
    except Exception as error:
        print(f"  {ticker}: fetch failed ({error})")
        return None

    frames = []
    for name, (tags, unit) in CONCEPTS.items():
        frame = concept_frame(facts, tags, unit, name)
        if frame is not None:
            frames.append(frame)

    if not frames:
        print(f"  {ticker}: no usable concepts")
        return None

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="end", how="outer")

    # Revenue and net income usually share a filing, but not always. The latest
    # of the filing dates is when every figure on the row was public.
    filed_columns = [column for column in merged.columns if column.startswith("filed_")]
    merged["filed"] = merged[filed_columns].max(axis=1)
    merged = merged.drop(columns=filed_columns)

    merged["ticker"] = ticker
    merged = merged.rename(columns={"end": "period_end"})

    return merged.sort_values("period_end")


def download_edgar(tickers, output_path=RAW_PATH / "edgar.parquet"):
    """Fetch every ticker's XBRL facts and cache them."""
    if "<your-email-here>" in USER_AGENT:
        raise RuntimeError(
            "Set USER_AGENT in edgar.py to include your email. The SEC blocks "
            "requests that do not identify a contact."
        )

    handle = session()
    cik_map = load_cik_map(handle)
    print(f"Loaded {len(cik_map)} ticker/CIK pairs")

    frames = []
    missing = []

    for n, ticker in enumerate(tickers, 1):
        cik = cik_map.get(ticker.upper())
        if cik is None:
            missing.append(ticker)
            continue

        frame = ticker_facts(handle, ticker, cik)
        if frame is not None:
            frames.append(frame)

        if n % 10 == 0:
            print(f"  {n}/{len(tickers)} tickers")
        time.sleep(REQUEST_DELAY)

    if missing:
        print(f"No CIK found for: {', '.join(missing)}")

    if not frames:
        raise RuntimeError("No EDGAR facts downloaded for any ticker.")

    raw = pd.concat(frames, ignore_index=True)
    raw.to_parquet(output_path, index=False)

    print(f"\nEDGAR: {len(raw)} ticker-quarters, {raw['ticker'].nunique()} tickers, "
          f"{raw['period_end'].min().date()} to {raw['period_end'].max().date()}")
    print(f"Quarters per ticker: median {int(raw.groupby('ticker').size().median())}")

    return raw


def build_edgar_features(raw):
    """Trailing-twelve-month figures, dated from the day they were filed."""
    raw = raw.sort_values(["ticker", "period_end"]).copy()
    grouped = raw.groupby("ticker")

    for column in ["revenue", "net_income", "eps"]:
        if column in raw.columns:
            raw[f"ttm_{column}"] = grouped[column].transform(
                lambda s: s.rolling(4, min_periods=4).sum()
            )

    if "ttm_net_income" in raw.columns and "ttm_revenue" in raw.columns:
        raw["profit_margin"] = (
            raw["ttm_net_income"] / raw["ttm_revenue"].replace(0, np.nan)
        )

    if "ttm_revenue" in raw.columns:
        raw["revenue_growth_yoy"] = grouped["ttm_revenue"].transform(
            lambda s: s / s.shift(4).replace(0, np.nan) - 1
        )

    # No lag estimate needed -- this is the date the filing actually landed.
    raw["available_from"] = raw["filed"]

    keep = ["ticker", "available_from", "profit_margin", "revenue_growth_yoy", "ttm_eps"]
    keep = [column for column in keep if column in raw.columns]

    return (
        raw[keep]
        .dropna(subset=["available_from"])
        .sort_values("available_from")
    )


def merge_edgar(df, features):
    """Attach each daily row the most recent filing that was public by then."""
    from fundamentals import align_keys

    df, features = align_keys(df, features, "date", "available_from")

    merged = pd.merge_asof(
        df, features,
        left_on="date", right_on="available_from",
        by="ticker", direction="backward",
    )

    # Price moves daily while the earnings figure holds, so this is built after
    # the join.
    if "ttm_eps" in merged.columns:
        merged["pe_ratio"] = merged["close"] / merged["ttm_eps"].replace(0, np.nan)

    return merged.drop(columns=["available_from", "ttm_eps"], errors="ignore")


def attach_edgar(df, path=RAW_PATH / "edgar.parquet"):
    """Merge EDGAR fundamentals if the cache exists, and report coverage."""
    if not path.exists():
        print(f"No {path} -- run edgar.py to fetch it")
        return df

    df = merge_edgar(df, build_edgar_features(pd.read_parquet(path)))

    present = [c for c in ["pe_ratio", "profit_margin", "revenue_growth_yoy"]
               if c in df.columns]
    if present:
        print("\nEDGAR coverage (share of rows with a value):")
        print(df[present].notna().mean().to_string(float_format=lambda v: f"{v:.3f}"))

    return df


if __name__ == "__main__":
    from extract_data import TICKERS

    download_edgar(TICKERS)
