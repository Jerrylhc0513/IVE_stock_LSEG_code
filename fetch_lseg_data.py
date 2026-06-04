"""
fetch_lseg_data.py — Fetch 1-minute LSEG data for 10 representative cross-sector U.S. stocks (7 months)

- Time range: 2025-01-01 → 2025-08-01 (7 months, aligned with the paper's sample length)
"""

import refinitiv.data as rd
import pandas as pd
import numpy as np
from pathlib import Path


# 10 representative large-cap U.S. stocks across sectors (selected by GICS sectors, aligned with the paper's top-100 selection idea)
RICS = [
    # Technology (2) - high-weight sector
    "AAPL.O",    # Apple - consumer electronics
    "NVDA.O",    # NVIDIA - semiconductor (high volatility)

    # Communication Services (1)
    "GOOGL.O",   # Alphabet - internet

    # Consumer Discretionary (2)
    "AMZN.O",    # Amazon - e-commerce
    "TSLA.O",    # Tesla - high-volatility retail stock

    # Consumer Staples (1)
    "COST.O",    # Costco - consumer staples / retail

    # Financials (1)
    "JPM.N",     # JPMorgan - banking leader

    # Health Care (1)
    "JNJ.N",     # Johnson & Johnson - healthcare leader

    # Industrials (1)
    "CAT.N",     # Caterpillar - industrial representative

    # Energy (1)
    "XOM.N",     # Exxon Mobil - energy representative
]
START_DATE = "2025-01-01"
END_DATE = "2025-08-01"
SAVE_DIR = Path("lseg_data")
MERGED_PATH = Path("all_stocks_1m.parquet")


def fetch_and_clean(ric: str) -> pd.DataFrame:
    print(f"\n[fetch] {ric} ...")
    df = rd.get_history(
        universe=ric, interval="1min",
        start=START_DATE, end=END_DATE,
        fields=["OPEN_PRC", "HIGH_1", "LOW_1", "TRDPRC_1",
                "ACVOL_UNS", "NUM_MOVES"],
    )
    if df is None or df.empty:
        print(f"[fetch] {ric}: no data returned, skipping")
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    df = df.rename(columns={
        "OPEN_PRC": "open", "HIGH_1": "high", "LOW_1": "low",
        "TRDPRC_1": "close", "ACVOL_UNS": "volume", "NUM_MOVES": "n_trades",
    })

    for col in ["open", "high", "low", "close", "volume", "n_trades"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("US/Eastern")

    rth_mask = (
        (df.index.time >= pd.Timestamp("09:30").time()) &
        (df.index.time < pd.Timestamp("16:00").time())
    )
    df = df[rth_mask]
    df = df.dropna(subset=["close"])
    df["date"] = df.index.date
    return df.reset_index()


def main():
    SAVE_DIR.mkdir(exist_ok=True)
    print("=" * 60)
    print("Step 1: Connect to LSEG")
    print("=" * 60)
    rd.open_session()
    print("Connected")

    print("\n" + "=" * 60)
    print(f"Step 2: Download {len(RICS)} stocks ({START_DATE} → {END_DATE})")
    print("=" * 60)
    print(f"Existing files will be skipped (delete the lseg_data directory first to refetch)")

    for ric in RICS:
        save_path = SAVE_DIR / f"{ric.replace('.', '_')}_1m.parquet"
        if save_path.exists():
            print(f"[skip] {save_path.name} already exists")
            continue
        df = fetch_and_clean(ric)
        if df is not None:
            df.to_parquet(save_path, index=False)
            print(f"[save] → {save_path}")

    try:
        rd.close_session()
    except Exception:
        pass

    print("\n" + "=" * 60)
    print("Step 3: Merge all stocks")
    print("=" * 60)
    all_dfs = []
    for ric in RICS:
        path = SAVE_DIR / f"{ric.replace('.', '_')}_1m.parquet"
        if not path.exists():
            print(f"[warn] {path.name} does not exist, skipping")
            continue
        df = pd.read_parquet(path)
        df["ric"] = ric
        all_dfs.append(df)
        print(f"  {ric:10s}  {len(df):>7} rows  {df['date'].nunique():>3} days")

    merged = pd.concat(all_dfs, ignore_index=True)
    merged.to_parquet(MERGED_PATH, index=False)

    print(f"\nMerge completed: {MERGED_PATH.resolve()}")
    print(f"   Total rows: {len(merged):,}")
    print(f"   Stocks: {merged['ric'].nunique()} stocks")
    print(f"   Dates: {merged['date'].nunique()} trading days")

    print("\nVolume statistics (should be positive, no NaN):")
    print(merged.groupby("ric")["volume"].agg(["min", "median", "max"]))


if __name__ == "__main__":
    main()