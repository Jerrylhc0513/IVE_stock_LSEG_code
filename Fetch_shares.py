"""
Fetch_shares.py — Fetch outstanding shares for each stock from LSEG

Purpose: Provide shares outstanding for data_lseg.py to calculate the true turnover_rate = volume / shares_outstanding
Note: Section 3.1 of the paper explicitly uses turnover rate, which is one of the four core IVE features
"""

import refinitiv.data as rd
import pandas as pd
from pathlib import Path


RICS = [
    "AAPL.O", "NVDA.O", "GOOGL.O", "AMZN.O", "TSLA.O",
    "COST.O", "JPM.N", "JNJ.N", "CAT.N", "XOM.N",
]
OUT_PATH = Path("shares_outstanding.parquet")


def main():
    print("[shares] Connecting to LSEG...")
    rd.open_session()

    print(f"[shares] Fetching shares outstanding for {len(RICS)} stocks...")
    df = rd.get_data(
        universe=RICS,
        fields=["TR.SharesOutstanding"],
    )
    print("[shares] Raw data:")
    print(df)
    print(f"\nColumns: {df.columns.tolist()}")

    # The column returned by LSEG is usually 'Outstanding Shares' but may vary
    # Use the non-Instrument column as the shares outstanding column
    shares_col = [c for c in df.columns if c != "Instrument"][0]
    df = df.rename(columns={
        "Instrument": "ric",
        shares_col: "shares_outstanding",
    })

    df["shares_outstanding"] = pd.to_numeric(
        df["shares_outstanding"], errors="coerce"
    )
    df = df.dropna(subset=["shares_outstanding"])
    df = df[["ric", "shares_outstanding"]].reset_index(drop=True)

    df.to_parquet(OUT_PATH, index=False)

    try:
        rd.close_session()
    except Exception:
        pass

    print(f"\nSaved to {OUT_PATH}")
    print(df)

if __name__ == "__main__":
    main()