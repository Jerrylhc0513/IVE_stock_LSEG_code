# IVE Replication — LSEG V2 (Faithful Reproduction)

This repository contains a PyTorch implementation of the **Intraday Volume
Estimator (IVE)** model proposed by Lee & Park (2024), trained on intraday
1-minute equity data from LSEG (London Stock Exchange Group, formerly
Refinitiv).

The pipeline reproduces the encoder–decoder Transformer architecture with a
Student's-t distribution head described in Section 3 of the original paper, and
extends it with a multi-step training objective and Time2Vec time encoding.

> **Paper reference**
> Lee, H. and Park, H. (2024).
> *IVE: Enhanced Probabilistic Forecasting of Intraday Volume Ratio with
> Transformers.* arXiv:2411.10956v1.

---

## 1. Project Structure

```
project/
├── fetch_lseg_data.py        # Pull 1-min OHLCV from LSEG
├── Fetch_shares.py            # Pull outstanding-shares reference data
├── data_lseg.py              # Feature engineering and PyTorch Dataset
├── model.py                  # PaperIVE model definition
├── train_paper.py            # Training loop (AdamW + cosine schedule)
├── Predict_2tower.py         # Inference and per-stock visualisation
├── all_stocks_1m.parquet      # (generated) merged 1-min data
├── shares_outstanding.parquet # (generated) shares outstanding per ticker
├── best_paper_5feat.pt        # (generated) best checkpoint by val NLL
├── prediction_metrics.txt     # (generated) per-stock MAE / RMSE / coverage
├── predictions_all_stocks.png # (generated) per-stock prediction plots
└── lseg_data/                 # (generated) per-ticker raw parquet files
```

---

## 2. Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0 (CUDA recommended)
- An LSEG Workspace / Eikon Data API session (`refinitiv-data` package)
- Sufficient GPU memory (the default 3M-parameter model fits comfortably on a
  12GB GPU at `batch_size=128`)

### Install

```bash
pip install torch numpy pandas pyarrow matplotlib refinitiv-data
```

### Authenticate to LSEG

A valid LSEG Workspace session must be running on the machine. The fetch
scripts call `rd.open_session()`, which uses the system-level LSEG
authentication.

---

## 3. Data

The pipeline is designed for U.S. equities. The default universe is **10
cross-sector large-cap stocks** selected via stratified GICS sampling:

| Sector | Tickers |
| --- | --- |
| Information Technology | AAPL, NVDA |
| Communication Services | GOOGL |
| Consumer Discretionary | AMZN, TSLA |
| Consumer Staples | COST |
| Financials | JPM |
| Health Care | JNJ |
| Industrials | CAT |
| Energy | XOM |

The default time range targets approximately 7 months ending at the time of
fetching. Note that LSEG academic-tier access limits 1-minute intraday history
to roughly the most recent 60 days, so the realised sample may be shorter than
the requested range. Adjust `START_DATE` and `END_DATE` in `fetch_lseg_data.py`
accordingly.

---

## 4. End-to-End Usage

The standard pipeline runs in five steps.

### Step 0 — Clean previous artefacts

If you have run the pipeline before and want a fresh start:

```bash
del all_stocks_1m.parquet
del shares_outstanding.parquet
del best_paper_5feat.pt
rmdir /s /q lseg_data
```

(On Linux/macOS replace `del` with `rm -f` and `rmdir /s /q` with `rm -rf`.)

### Step 1 — Fetch 1-minute OHLCV

```bash
python fetch_lseg_data.py
```

This downloads one parquet file per ticker into `lseg_data/`, then merges them
into a single `all_stocks_1m.parquet`. Already-downloaded tickers are skipped;
delete the `lseg_data/` directory to force a re-fetch.

### Step 2 — Fetch shares outstanding

```bash
python Fetch_shares.py
```

This calls the LSEG reference API to retrieve the most recent shares
outstanding for each ticker, which is required to compute the paper's
*turnover rate* feature.

### Step 3 — Train the model

```bash
python train_paper.py --epochs 12
```

Common flags:

| Flag | Default | Notes |
| --- | --- | --- |
| `--epochs` | 12 | Number of training epochs |
| `--batch_size` | 128 | Per-step minibatch size |
| `--d_model` | 192 | Transformer hidden dim |
| `--n_layers` | 4 | Number of encoder and decoder layers (paper default) |
| `--multi_step` | 3 | Multi-step prediction horizon (paper default) |
| `--lr` | 3e-4 | AdamW learning rate (paper default) |
| `--save_path` | best_paper_5feat.pt | Checkpoint path |

The script saves the checkpoint with the lowest validation NLL.

### Step 4 — Run inference and produce plots

```bash
python Predict_2tower.py
```

This loads the best checkpoint, evaluates on the validation set, prints
per-stock and global metrics, writes them to `prediction_metrics.txt`, and
saves a multi-row figure to `predictions_all_stocks.png` showing predicted
mean ±1σ vs target for each stock.

---

## 5. Features

The model uses **5 input features** that map directly to the four core
quantities described in Section 3.1 of the original paper, plus an absolute
intraday time index:

| # | Name | Paper variable |
| --- | --- | --- |
| 1 | `log_volume` | Volume |
| 2 | `log_acc_volume` | Accumulated volume |
| 3 | `log_turnover` | Turnover rate (volume / shares outstanding) |
| 4 | `log_amount` | Trade amount (price × volume) |
| 5 | `minute_idx_float` | Absolute intraday time information |

Stock identity is handled separately via a learned `nn.Embedding` indexed by
`stock_id`, matching the paper's *stock-specific categorical information*.
Sinusoidal positional encoding and a learnable Time2Vec module provide the
*Time Encoding alongside the standard sinusoidal positional embedding*
described in the paper.

The prediction target is the logarithm of the intraday volume ratio,
clipped to `[-10, 2]` to suppress numerical instability arising from
near-zero volume minutes.

---

## 6. Model

`PaperIVE` is a two-tower encoder–decoder Transformer:

- **Encoder** — 4 layers (pre-norm, batch-first), consumes the full 390-minute
  previous trading day.
- **Decoder** — 4 layers, consumes the partially-observed current day up to
  minute *t*, with causal masking.
- **Distribution head** — three linear projections producing `mu`, `sigma`,
  and `nu` of a Student's-t distribution at each of the next `multi_step`
  time points.
- **Loss** — Student's-t negative log-likelihood, averaged across batch and
  multi-step targets.

Approximate parameter count: ~3 million at the default configuration.

---

## 7. Evaluation Protocol

Following the paper's convention, training uses multi-step targets to prevent
the trivial copy solution, but **inference and reported metrics use only the
first-step prediction**. Metrics computed in `Predict_2tower.py`:

- `MAE` — Mean Absolute Error of the predicted mean
- `RMSE` — Root Mean Square Error of the predicted mean
- `1σ coverage` — Fraction of targets falling within `mu ± sigma`

Both per-stock and global aggregates are reported.

---

## 8. Known Caveats

- **Sample size.** The pipeline as configured produces samples roughly two
  orders of magnitude smaller than the original paper's training set
  (~5M samples per market). MAE figures should be interpreted accordingly.
- **Data source asymmetry.** Some non-technology stocks (e.g. CAT, XOM, JNJ)
  occasionally have minutes with zero trades. The pipeline keeps days with at
  least 380 observed minutes and forward-fills prices on the missing minutes
  with zero volume.
- **Initialisation variance.** With small training sets, identical
  configurations can produce noticeably different final MAE across random
  seeds. Multi-seed averaging is recommended for any quoted result.
- **LSEG retention window.** Academic-tier LSEG 1-minute data is typically
  available for the most recent ~60 days only, regardless of the requested
  date range. Verify the actual range returned by `fetch_lseg_data.py` before
  drawing conclusions about long-range stability.

---
