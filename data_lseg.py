import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


PARQUET_PATH = Path("all_stocks_1m.parquet")
SHARES_PATH = Path("shares_outstanding.parquet")
FULL_DAY_LEN = 390  # 390 minutes in one regular trading day, consistent with the paper's context length

FEATURE_COLS = [
    "log_volume",                       # 1.  Paper core feature: volume
    "log_volume_ratio",                 # 2.  Historical value of the prediction target
    "acc_volume_ratio",                 # 3.  Paper core feature: accumulated volume
    "log_amount",                       # 4.  Paper core feature: trade amount
    "log_n_trades",                     # 5.  Number of trades / activity count
    "log_turnover",                     # 6.  Paper core feature: turnover rate
    "turnover_proxy",                   # 7.  Relative historical activity, complementary to #6
    "price_change",                     # 8.  Price signal
    "lag_1_log_volume_ratio",           # 9.  Previous minute
    "lag_yesterday_log_volume_ratio",   # 10. Previous day same-minute value, strongest signal
    "minute_sin",                       # 11. Paper sinusoidal positional feature
    "minute_cos",                       # 12. Paper sinusoidal positional feature
    "dow_sin",                          # 13. Day of week
    "dow_cos",                          # 14. Day of week
    "day_idx_float",                    # 15. Used for Time2Vec, paper time encoding
]
TARGET_COL = "log_volume_ratio"


# ============================================================
# Feature engineering
# ============================================================

def add_features(df: pd.DataFrame, shares_df: pd.DataFrame = None) -> pd.DataFrame:
    df = df.copy()

    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    else:
        raise ValueError("Missing Timestamp column")

    df = df.sort_values(["ric", "Timestamp"]).reset_index(drop=True)

    # ---------- Basic volume / amount features ----------
    df["log_volume"] = np.log1p(df["volume"])
    df["log_n_trades"] = np.log1p(df["n_trades"])

    # volume_ratio + log_volume_ratio (prediction target)
    daily_total = df.groupby(["ric", "date"])["volume"].transform("sum")
    df["volume_ratio"] = df["volume"] / daily_total.clip(lower=1e-9)
    df["log_volume_ratio"] = np.log(df["volume_ratio"].clip(lower=1e-12))

    # accumulated volume ratio (paper accumulated volume, cross-stock comparable version)
    df["acc_volume"] = df.groupby(["ric", "date"])["volume"].cumsum()
    df["acc_volume_ratio"] = df["acc_volume"] / daily_total.clip(lower=1e-9)

    # trade amount (paper trade amount)
    df["amount"] = df["close"] * df["volume"]
    df["log_amount"] = np.log1p(df["amount"])

    # ---------- True turnover using shares outstanding ----------
    if shares_df is not None:
        df = df.merge(shares_df, on="ric", how="left")
        df["turnover_rate"] = df["volume"] / df["shares_outstanding"].clip(lower=1)
        df["log_turnover"] = np.log1p(df["turnover_rate"] * 1e6)
    else:
        print("[features] Warning: shares_outstanding.parquet is missing; log_turnover is filled with 0")
        df["log_turnover"] = 0.0

    # ---------- turnover_proxy: relative historical activity ----------
    rolling_mean = df.groupby("ric")["volume"].transform(
        lambda x: x.rolling(window=FULL_DAY_LEN * 30, min_periods=FULL_DAY_LEN).mean()
    )
    df["turnover_proxy"] = np.log(
        df["volume"].clip(lower=1) / rolling_mean.clip(lower=1)
    ).fillna(0)

    # ---------- Price features ----------
    daily_open = df.groupby(["ric", "date"])["open"].transform("first")
    df["price_change"] = (df["close"] - daily_open) / daily_open.clip(lower=1e-6)

    # ---------- Time encoding ----------
    df["minute_idx"] = df.groupby(["ric", "date"]).cumcount()
    df["minute_sin"] = np.sin(2 * np.pi * df["minute_idx"] / FULL_DAY_LEN)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute_idx"] / FULL_DAY_LEN)

    df["dow"] = df["Timestamp"].dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

    # day_idx_float: used for Time2Vec
    all_dates = sorted(df["date"].unique())
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    df["day_idx_float"] = df["date"].map(date_to_idx).astype(np.float32)
    df["day_idx_float"] /= max(len(all_dates) - 1, 1)

    # ---------- Lag features ----------
    df["lag_1_log_volume_ratio"] = (
        df.groupby(["ric", "date"])["log_volume_ratio"].shift(1).fillna(0)
    )

    df = df.sort_values(["ric", "date", "minute_idx"]).reset_index(drop=True)
    df["lag_yesterday_log_volume_ratio"] = (
        df.groupby(["ric", "minute_idx"])["log_volume_ratio"].shift(1)
    )
    df["lag_yesterday_log_volume_ratio"] = (
        df["lag_yesterday_log_volume_ratio"].fillna(df["log_volume_ratio"].mean())
    )

    return df


def filter_full_days(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only stock-date pairs with exactly 390 rows."""
    day_counts = df.groupby(["ric", "date"]).size()
    full_days = day_counts[day_counts == FULL_DAY_LEN].index
    df_full = df.set_index(["ric", "date"]).loc[full_days].reset_index()
    return df_full


# ============================================================
# Dataset with multi-step prediction support
# ============================================================

class IVEPaperDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 stock_to_id: dict,
                 min_t: int = 5,
                 max_t: int = FULL_DAY_LEN - 3,
                 feature_mean: np.ndarray = None,
                 feature_std: np.ndarray = None,
                 multi_step: int = 3):
        self.stock_to_id = stock_to_id
        self.min_t = min_t
        self.max_t = max_t
        self.multi_step = multi_step

        feats = df[FEATURE_COLS].values.astype(np.float32)
        if feature_mean is None:
            self.mean = feats.mean(axis=0)
            self.std = feats.std(axis=0) + 1e-6
        else:
            self.mean = feature_mean
            self.std = feature_std

        self.day_arrays = {}
        self.day_targets = {}
        for (ric, date), g in df.groupby(["ric", "date"]):
            g = g.sort_values("minute_idx")
            self.day_arrays[(ric, date)] = (
                (g[FEATURE_COLS].values.astype(np.float32) - self.mean) / self.std
            )
            self.day_targets[(ric, date)] = g[TARGET_COL].values.astype(np.float32)

        sorted_dates_by_ric = (
            df.groupby("ric")["date"].unique().apply(sorted).to_dict()
        )

        self.samples = []
        for ric, dates in sorted_dates_by_ric.items():
            for i in range(1, len(dates)):
                date_y = dates[i - 1]
                date_t = dates[i]
                for t in range(min_t, max_t + 1):
                    self.samples.append((ric, date_y, date_t, t))

        print(f"[Dataset] Number of samples: {len(self.samples)}")
        print(f"[Dataset] Feature dimension: {len(FEATURE_COLS)}")
        print(f"[Dataset] multi_step: {multi_step}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ric, date_y, date_t, t = self.samples[idx]
        yesterday = self.day_arrays[(ric, date_y)]
        today_full = self.day_arrays[(ric, date_t)]
        today = today_full[:t]

        targets_arr = self.day_targets[(ric, date_t)]
        if t + self.multi_step <= len(targets_arr):
            targets = targets_arr[t:t + self.multi_step]
        else:
            targets = targets_arr[t:]
            pad_size = self.multi_step - len(targets)
            if pad_size > 0:
                targets = np.concatenate([targets, np.full(pad_size, targets[-1])])

        stock_id = self.stock_to_id[ric]

        return {
            "yesterday": torch.from_numpy(yesterday),
            "today": torch.from_numpy(today),
            "today_len": t,
            "day_idx": float(t),
            "target": torch.from_numpy(targets.astype(np.float32)),
            "stock_id": torch.tensor(stock_id, dtype=torch.long),
        }


# ============================================================
# Batch collate
# ============================================================

def collate_fn(batch):
    yesterdays = torch.stack([b["yesterday"] for b in batch])
    today_lens = torch.tensor([b["today_len"] for b in batch], dtype=torch.long)
    max_t = today_lens.max().item()

    F_dim = yesterdays.shape[-1]
    B = len(batch)
    todays_padded = torch.zeros(B, max_t, F_dim, dtype=torch.float32)
    today_mask = torch.zeros(B, max_t, dtype=torch.bool)

    for i, b in enumerate(batch):
        L = b["today_len"]
        todays_padded[i, :L] = b["today"]
        today_mask[i, :L] = True

    targets = torch.stack([b["target"] for b in batch])
    stock_ids = torch.stack([b["stock_id"] for b in batch])
    day_idx = torch.tensor([b["day_idx"] for b in batch], dtype=torch.float32)

    return {
        "yesterday": yesterdays,
        "today": todays_padded,
        "today_mask": today_mask,
        "day_idx": day_idx,
        "target": targets,
        "stock_id": stock_ids,
    }


# ============================================================
# DataLoader preparation
# ============================================================

def prepare_dataloaders(
    parquet_path: Path = PARQUET_PATH,
    shares_path: Path = SHARES_PATH,
    batch_size: int = 128,
    val_ratio: float = 0.15,
    min_t: int = 5,
    max_t: int = FULL_DAY_LEN - 3,
    multi_step: int = 3,
):
    print(f"[data] Reading {parquet_path}")
    df_raw = pd.read_parquet(parquet_path)
    print(f"[data] Raw data: {df_raw.shape}, {df_raw['ric'].nunique()} stocks")

    shares_df = None
    if shares_path.exists():
        shares_df = pd.read_parquet(shares_path)
        print(f"[data] Loaded shares outstanding:")
        for _, row in shares_df.iterrows():
            print(f"        {row['ric']:10s}  {int(row['shares_outstanding']):,}")
    else:
        print(f"[data] Warning: {shares_path} not found; skipping true turnover")

    df_feat = add_features(df_raw, shares_df)
    df_full = filter_full_days(df_feat)
    print(f"[data] Full 390-row trading days: {df_full.groupby('ric')['date'].nunique().to_dict()}")

    all_dates = sorted(df_full["date"].unique())
    cut = int(len(all_dates) * (1 - val_ratio))
    train_dates = set(all_dates[:cut])
    val_dates = set(all_dates[cut:])
    print(f"[data] Training dates: {len(train_dates)} days, validation dates: {len(val_dates)} days")

    train_df = df_full[df_full["date"].isin(train_dates)].reset_index(drop=True)
    val_df = df_full[df_full["date"].isin(val_dates)].reset_index(drop=True)

    stocks = sorted(df_full["ric"].unique())
    stock_to_id = {s: i for i, s in enumerate(stocks)}

    train_ds = IVEPaperDataset(train_df, stock_to_id, min_t, max_t,
                               multi_step=multi_step)
    val_ds = IVEPaperDataset(val_df, stock_to_id, min_t, max_t,
                             feature_mean=train_ds.mean, feature_std=train_ds.std,
                             multi_step=multi_step)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          collate_fn=collate_fn, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)

    return train_dl, val_dl, train_ds, stock_to_id


# ============================================================
# Self-check
# ============================================================

if __name__ == "__main__":
    train_dl, val_dl, train_ds, stock_to_id = prepare_dataloaders(
        batch_size=8, val_ratio=0.15, multi_step=3,
    )

    print("\n=== Test one batch ===")
    batch = next(iter(train_dl))
    for k, v in batch.items():
        if torch.is_tensor(v):
            print(f"  {k:12s}: shape={tuple(v.shape)}, dtype={v.dtype}")
        else:
            print(f"  {k:12s}: {v}")

    print(f"\nFeature list ({len(FEATURE_COLS)} dimensions):")
    for i, c in enumerate(FEATURE_COLS):
        print(f"  [{i:2d}] {c}")

    print("\nSelf-check passed")