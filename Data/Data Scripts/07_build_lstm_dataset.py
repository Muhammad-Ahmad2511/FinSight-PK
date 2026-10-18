"""
Build the LSTM datasets for KSE-100 forecasting.

Step 1-4: merge price + indicators + regime, EPU and news sentiment onto the
          KSE-100 trading calendar (same logic as before).
Step 5  : feature cleanup + rolling sentiment features.
Step 6  : forecasting targets for 3, 7 and 14 trading days ahead, with a
          chronological train/val/test split column.

Outputs (all in DATA_DIR):
  kse100_lstm_merged.csv   merged features, no targets (previous output)
  kse100_lstm_features.csv engineered model features for ALL rows, no targets
                           (use the last 30 rows to forecast beyond the last target)
  kse100_lstm_h3.csv       features + 3-day targets
  kse100_lstm_h7.csv       features + 7-day targets
  kse100_lstm_h14.csv      features + 14-day targets
  kse100_lstm_multi.csv    features + all three horizons (for one multi-output model)

Targets (per horizon h):
  target_close_h{h}   Close at t+h (index points)
  target_logret_h{h}  log(Close[t+h] / Close[t])  <- train on this, not raw Close
  target_date_h{h}    date of t+h
`split` is one of train / val / test / purge. Never sample from 'purge' rows.
Convert a predicted log-return back with: Close[t] * exp(pred).

Run (Colab): !python 07_build_lstm_dataset.py      (reads/writes /content)
Run (local): python 07_build_lstm_dataset.py       (reads/writes the current folder)
Set the env var FYP_DATA_DIR to override the folder.
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path

# --- Config -- all files live flat in this one folder ---
# Colab: upload the 3 input CSVs to /content (default). Local: set FYP_DATA_DIR or edit the fallback.
_default = "/content" if Path("/content").exists() else "."
DATA_DIR = Path(os.environ.get("FYP_DATA_DIR", _default))
INPUT_KSE_REGIME = DATA_DIR / "kse100_with_regime.csv"        # already includes indicators
INPUT_EPU = DATA_DIR / "pakistan_epu.csv"
INPUT_NEWS = DATA_DIR / "dawn_business_2015_to_2026_labeled.csv"
OUTPUT_MERGED = DATA_DIR / "kse100_lstm_merged.csv"

HORIZONS = (3, 7, 14)                 # trading days ahead
SPLIT = (0.70, 0.15, 0.15)            # chronological train / val / test
PURGE_BOUNDARIES = True               # label the last h rows before each split boundary 'purge' so
                                      # train/val targets don't reach into the next split. The rows
                                      # are KEPT (they are needed as contiguous input context) but
                                      # must not be used as training/eval samples.
DROP_VOLUME = True                    # 40% missing, concentrated in 2019-2024, not random
DROP_REDUNDANT = True                 # drop `Regime` string (one-hots kept) and `Change` (= Return*100)

# =============================================================================
# 1. Base trading calendar: price + technical indicators + regime
# =============================================================================
kse = pd.read_csv(INPUT_KSE_REGIME)
kse["Date"] = pd.to_datetime(kse["Date"])
kse = kse.sort_values("Date").reset_index(drop=True)

if DROP_VOLUME and "Volume" in kse.columns:
    n_missing = kse["Volume"].isna().sum()
    print(f"Dropping Volume: {n_missing}/{len(kse)} rows missing ({n_missing/len(kse):.0%}), "
          f"concentrated in 2019-2024 (not random gaps). Set DROP_VOLUME=False to keep it.")
    kse = kse.drop(columns=["Volume"])

# =============================================================================
# 2. Macroeconomic uncertainty (EPU) -- already daily, direct join on Date
# =============================================================================
epu = pd.read_csv(INPUT_EPU)
epu["Date"] = pd.to_datetime(epu["Date"])
epu = epu.drop_duplicates(subset="Date").sort_values("Date")

merged = kse.merge(epu, on="Date", how="left")

# =============================================================================
# 3. News sentiment -- aggregate per CALENDAR day, then roll weekend/holiday
#    news forward into the next trading day.
# =============================================================================
news = pd.read_csv(INPUT_NEWS)
news["date"] = pd.to_datetime(news["date"])

# continuous per-article score in [-1, 1] from the class probabilities
news["sentiment_score"] = news["prob_positive"] - news["prob_negative"]

# label casing differs between files ("Positive" vs "positive"), so normalise
lbl = news["distilbert_label"].str.lower()

daily_news = (
    news.assign(_pos=(lbl == "positive"), _neg=(lbl == "negative"), _neu=(lbl == "neutral"))
    .groupby("date")
    .agg(
        sentiment_mean=("sentiment_score", "mean"),
        sentiment_sum=("sentiment_score", "sum"),
        article_count=("sentiment_score", "count"),
        positive_count=("_pos", "sum"),
        negative_count=("_neg", "sum"),
        neutral_count=("_neu", "sum"),
        confidence_mean=("distilbert_confidence", "mean"),
    )
    .reset_index()
    .rename(columns={"date": "Date"})
)

# map every calendar day to the next trading day on/after it
trading_days = merged["Date"].sort_values().to_numpy()
all_days = pd.date_range(daily_news["Date"].min(), merged["Date"].max(), freq="D")
bucket_end = pd.Series(
    np.searchsorted(trading_days, all_days.values, side="left"),
    index=all_days,
)
bucket_end = bucket_end.clip(upper=len(trading_days) - 1).map(lambda i: trading_days[i])
day_to_trading_day = bucket_end.to_dict()

daily_news["Date"] = daily_news["Date"].map(day_to_trading_day)

# mean of calendar-day means, as in report Sec. 5.3.3 (matches the existing LSTM_Dataset.csv)
sent_agg = daily_news.groupby("Date").agg(
    sentiment_mean=("sentiment_mean", "mean"),
    sentiment_sum=("sentiment_sum", "sum"),
    article_count=("article_count", "sum"),
    positive_count=("positive_count", "sum"),
    negative_count=("negative_count", "sum"),
    neutral_count=("neutral_count", "sum"),
    confidence_mean=("confidence_mean", "mean"),
).reset_index()

merged = merged.merge(sent_agg, on="Date", how="left")

# no news on a trading day -> zeros (article_count==0 flags it, see has_news below)
sentiment_cols = ["sentiment_mean", "sentiment_sum", "article_count",
                  "positive_count", "negative_count", "neutral_count", "confidence_mean"]
for col in sentiment_cols:
    merged[col] = merged[col].fillna(0)

# =============================================================================
# 4. Drop rows that can't have complete features (pre-EPU, indicator warm-up)
# =============================================================================
before = len(merged)
merged = merged.dropna(subset=["EPU", "RSI_14", "Return", "Roll_Return_20", "Roll_Vol_20", "Regime"])
merged = merged.reset_index(drop=True)
print(f"Dropped {before - len(merged)} warm-up/pre-EPU rows ({before} -> {len(merged)})")

merged.to_csv(OUTPUT_MERGED, index=False)
print(f"Saved merged features: {OUTPUT_MERGED.name} ({merged.shape[0]} x {merged.shape[1]})")

# =============================================================================
# 5. Feature cleanup + rolling sentiment (past-only, no look-ahead)
# =============================================================================
feat = merged.copy()

# regime one-hots (create if the input file only has the string label)
for r in ["Bear", "Bull", "Sideways"]:
    col = f"Regime_{r}"
    if col not in feat.columns:
        feat[col] = (feat["Regime"].str.title() == r)
    feat[col] = feat[col].astype(int)

feat["has_news"] = (feat["article_count"] > 0).astype(int)
feat["sentiment_mean_3d"] = feat["sentiment_mean"].rolling(3, min_periods=1).mean()
feat["sentiment_mean_7d"] = feat["sentiment_mean"].rolling(7, min_periods=1).mean()

if DROP_REDUNDANT:
    feat = feat.drop(columns=[c for c in ["Regime", "Change"] if c in feat.columns])

# =============================================================================
# 6. Targets + chronological split
# =============================================================================
def add_targets(df, horizons):
    out = df.copy()
    for h in horizons:
        out[f"target_close_h{h}"] = out["Close"].shift(-h)
        out[f"target_logret_h{h}"] = np.log(out[f"target_close_h{h}"] / out["Close"])
        out[f"target_date_h{h}"] = out["Date"].shift(-h)
    # the last max(h) rows have no future close -> drop
    tcols = [f"target_close_h{h}" for h in horizons]
    return out.dropna(subset=tcols).reset_index(drop=True)


def add_split(df, max_h):
    n = len(df)
    i_val = int(n * SPLIT[0])
    i_test = int(n * (SPLIT[0] + SPLIT[1]))
    split = np.array(["train"] * n, dtype=object)
    split[i_val:i_test] = "val"
    split[i_test:] = "test"
    df = df.copy()
    df["split"] = split
    if PURGE_BOUNDARIES:
        for b in (i_val, i_test):
            df.loc[max(0, b - max_h):b - 1, "split"] = "purge"
    return df


def report(name, df, h_list):
    print(f"\n{name}: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"  purge {(df.split == 'purge').sum():5d} rows (context only)")
    for s in ["train", "val", "test"]:
        d = df[df.split == s]
        print(f"  {s:5s} {len(d):5d} rows  {d['Date'].min().date()} -> {d['Date'].max().date()}  "
              f"Close {d['Close'].min():,.0f}-{d['Close'].max():,.0f}")
    for h in h_list:
        lr = df[f"target_logret_h{h}"]
        print(f"  h={h:2d} log-return target: mean {lr.mean():+.4f}  std {lr.std():.4f}")
    nan = df.isna().sum()
    print("  NaNs:", "none" if nan.sum() == 0 else nan[nan > 0].to_dict())


feat.to_csv(DATA_DIR / "kse100_lstm_features.csv", index=False)
print(f"Saved features-only file: kse100_lstm_features.csv ({feat.shape[0]} x {feat.shape[1]})")

# one file per horizon
for h in HORIZONS:
    d = add_split(add_targets(feat, [h]), max_h=h)
    path = DATA_DIR / f"kse100_lstm_h{h}.csv"
    d.to_csv(path, index=False)
    report(path.name, d, [h])

# one file with all horizons (multi-output model)
multi = add_split(add_targets(feat, HORIZONS), max_h=max(HORIZONS))
path = DATA_DIR / "kse100_lstm_multi.csv"
multi.to_csv(path, index=False)
report(path.name, multi, HORIZONS)

feature_cols = [c for c in multi.columns
                if not c.startswith("target_") and c not in ("Date", "split")]
print(f"\nInput features ({len(feature_cols)}): {feature_cols}")
print("\nNotes for training:")
print("  - rows with split=='purge' are input context only; sample windows only from train/val/test rows")
print("  - fit scalers on rows where split == 'train' only")
print("  - train on target_logret_h*, convert back with Close[t] * exp(pred)")
