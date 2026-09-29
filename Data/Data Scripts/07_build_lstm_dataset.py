"""
Merge price+indicators+regime, EPU, and news sentiment into one LSTM-ready
dataset, aligned to the KSE-100 trading calendar.

Run locally: python build_lstm_dataset.py
Edit the four INPUT_* paths below to match your repo.
"""
import pandas as pd
import numpy as np
from pathlib import Path

# --- Config -- all files live flat in this one folder ---
DATA_DIR = Path(r"C:\Users\Win 1o pro\Downloads\fyp")
INPUT_KSE_REGIME = DATA_DIR / "kse100_with_regime.csv"        # already includes indicators
INPUT_EPU = DATA_DIR / "pakistan_epu.csv"
INPUT_NEWS = DATA_DIR / "dawn_business_2015_to_2026_labeled.csv"
OUTPUT_PATH = DATA_DIR / "kse100_lstm_merged.csv"

DROP_VOLUME = True   # see note printed below -- 40% missing, concentrated in 2019-2024, not random

# =============================================================================
# 1. Base trading calendar: price + technical indicators + regime
# =============================================================================
kse = pd.read_csv(INPUT_KSE_REGIME)
kse["Date"] = pd.to_datetime(kse["Date"])
kse = kse.sort_values("Date").reset_index(drop=True)

if DROP_VOLUME:
    n_missing = kse["Volume"].isna().sum()
    print(f"Dropping Volume: {n_missing}/{len(kse)} rows missing ({n_missing/len(kse):.0%}), "
          f"concentrated in 2019-2024 (not random gaps) -- unreliable as an LSTM feature. "
          f"Set DROP_VOLUME=False to keep it (you'll need to handle the block-missingness yourself).")
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
#    news forward into the next trading day (news publishes 7 days/week, the
#    market doesn't -- a Saturday/Sunday headline should still inform Monday).
# =============================================================================
news = pd.read_csv(INPUT_NEWS)
news["date"] = pd.to_datetime(news["date"])

# continuous per-article sentiment score in [-1, 1]; using the softmax
# probabilities (not just the argmax label) keeps low-confidence rows from
# swinging the daily average as hard as high-confidence ones
news["sentiment_score"] = news["prob_positive"] - news["prob_negative"]

daily_news = news.groupby("date").agg(
    sentiment_mean=("sentiment_score", "mean"),
    sentiment_sum=("sentiment_score", "sum"),
    article_count=("sentiment_score", "count"),
    positive_count=("distilbert_label", lambda s: (s == "Positive").sum()),
    negative_count=("distilbert_label", lambda s: (s == "Negative").sum()),
    neutral_count=("distilbert_label", lambda s: (s == "Neutral").sum()),
    confidence_mean=("distilbert_confidence", "mean"),
).reset_index().rename(columns={"date": "Date"})

# build a bucket_end for every calendar day = the next trading day on/after it
trading_days = merged["Date"].sort_values().to_numpy()
all_days = pd.date_range(daily_news["Date"].min(), merged["Date"].max(), freq="D")
bucket_end = pd.Series(
    np.searchsorted(trading_days, all_days.values, side="left"),
    index=all_days,
)
bucket_end = bucket_end.clip(upper=len(trading_days) - 1).map(lambda i: trading_days[i])
day_to_trading_day = bucket_end.to_dict()

daily_news["Date"] = daily_news["Date"].map(day_to_trading_day)

# now sum/mean across all calendar days that rolled into the same trading day
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

# a trading day with genuinely zero rolled-in news (rare) gets neutral/zero,
# not NaN -- NaN would break an LSTM, and "no news" is itself real information
sentiment_cols = ["sentiment_mean", "sentiment_sum", "article_count",
                   "positive_count", "negative_count", "neutral_count", "confidence_mean"]
for col in sentiment_cols:
    merged[col] = merged[col].fillna(0)

# =============================================================================
# 4. Drop rows that can't have complete features:
#    - before EPU coverage starts (2014 warm-up)
#    - before the technical-indicator warm-up window (RSI_14/Roll_*/Regime)
# =============================================================================
before = len(merged)
merged = merged.dropna(subset=["EPU", "RSI_14", "Return", "Roll_Return_20", "Roll_Vol_20", "Regime"])
merged = merged.reset_index(drop=True)
print(f"Dropped {before - len(merged)} warm-up/pre-EPU rows ({before} -> {len(merged)})")

# =============================================================================
# 5. Save + sanity report
# =============================================================================
merged.to_csv(OUTPUT_PATH, index=False)

print(f"\nFinal merged dataset: {merged.shape[0]} rows x {merged.shape[1]} columns")
print(f"Date range: {merged['Date'].min().date()} -> {merged['Date'].max().date()}")
print("Remaining NaNs per column (should all be 0):")
nan_counts = merged.isna().sum()
print(nan_counts[nan_counts > 0] if nan_counts.sum() > 0 else "  none")
print(f"\nSaved to: {OUTPUT_PATH.resolve()}")
