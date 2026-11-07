"""
FinSight PK - Step 7: build the LSTM datasets for KSE-100 forecasting.

Merges, on the KSE-100 trading calendar starting 2013-02-02:
  price + technical indicators + regime   (kse100_with_regime.csv)
  official EPU + Dawn EPU features        (pakistan_epu.csv, from 04_epu_clean.py)
  external market / macro features        (external_features.csv, from 06_fetch_external.py, optional)
  Dawn news sentiment                     (labeled news CSV(s), optional files merged + de-duplicated)

Outputs (all in DATA_DIR):
  kse100_lstm_merged.csv    merged features, no targets
  kse100_lstm_features.csv  engineered model features for ALL rows, no targets
  kse100_lstm_h{h}.csv      features + targets for each horizon
  kse100_lstm_multi.csv     features + all horizons (multi-output model)

Targets (per horizon h, in trading days):
  target_close_h{h}, target_logret_h{h} = log(Close[t+h]/Close[t]), target_date_h{h}
`split` is train / val / test / purge. Never sample windows ending in 'purge' rows.

Config via env vars:
  FYP_DATA_DIR     folder with inputs/outputs (default: /content on Colab, else script folder)
  FYP_START_DATE   first date kept (default 2013-02-02)
  FYP_HORIZONS     comma list (default 1,3,5)
  FYP_NEWS_FILES   comma list of labeled news CSVs (default dawn_business_2015_to_2026_labeled.csv)

New sentiment features (computed from relevance-filtered articles, past-only):
  sent_surprise    z-score of today's mean sentiment vs the previous 30 trading days (no-news days excluded)
  sent_ewm_hl1/3/10  exponentially weighted sentiment, half-lives 1, 3, 10 trading days
  neg_share_7d     share of negative articles over the last 7 trading days (pooled)
  count_z          z-score of log(1+relevant article count) vs previous 30 trading days
Compact set for ablation: sent_surprise, sent_ewm_hl3, neg_share_7d, count_z.
The old 10 sentiment columns are kept for comparison.

Run:  python 07_build_lstm_dataset.py
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path

_default = "/content" if Path("/content").exists() else str(Path(__file__).resolve().parent)
DATA_DIR = Path(os.environ.get("FYP_DATA_DIR", _default))

START_DATE = pd.Timestamp(os.environ.get("FYP_START_DATE", "2013-02-02"))
HORIZONS = tuple(int(x) for x in os.environ.get("FYP_HORIZONS", "1,3,5").split(","))
NEWS_FILES = [s.strip() for s in os.environ.get(
    "FYP_NEWS_FILES", "dawn_business_2013_to_2026_labeled.csv").split(",") if s.strip()]

# markets/macro relevance filter on title + lead text (set FYP_RELEVANCE=0 to keep all articles)
USE_RELEVANCE = os.environ.get("FYP_RELEVANCE", "1") != "0"
RELEVANCE_PATTERNS = [r"\bstock", r"\bkse\b", r"\bkse-?100\b", r"\bpsx\b", r"\bbourse\b", r"\bequit(?:y|ies)\b",
    r"\bshares?\b", r"\bindex\b", r"\beconom\w*", r"\binflation\b", r"\bcpi\b", r"\bgdp\b",
    r"\binterest rates?\b", r"\bpolicy rate\b", r"\bsbp\b", r"\bstate bank\b", r"\bcentral bank\b",
    r"\bmonetary\b", r"\bfiscal\b", r"\bbudget\b", r"\bimf\b", r"\brupee\b", r"\bpkr\b", r"\bdollar\b",
    r"\bexchange rate\b", r"\bcurrency\b", r"\bremittances?\b", r"\bforex\b", r"\breserves\b",
    r"\bbond\b", r"\bt-?bills?\b", r"\bearnings\b", r"\bprofit\b", r"\bdividend\b", r"\binvestors?\b",
    r"\bmarket\b", r"\bdebt\b", r"\bexports?\b", r"\bimports?\b", r"\btrade deficit\b", r"\bcurrent account\b"]
SURPRISE_WINDOW = 30

def parse_dates(col):
    """Per-row strict parsing: YYYY-MM-DD (ISO) or DD-MM-YY (labeled news file). Anything else -> NaT."""
    s = col.astype(str).str.strip().str.slice(0, 10)
    iso = s.str.match(r"^\d{4}-\d{2}-\d{2}$")
    dmy = s.str.match(r"^\d{2}-\d{2}-\d{2}$")
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    out[iso] = pd.to_datetime(s[iso], format="%Y-%m-%d", errors="coerce")
    out[dmy] = pd.to_datetime(s[dmy], format="%d-%m-%y", errors="coerce")
    print(f"  date parsing: {int(iso.sum())} YYYY-MM-DD, {int(dmy.sum())} DD-MM-YY, "
          f"{int((~iso & ~dmy).sum())} unrecognised {s[~iso & ~dmy].head(3).tolist()}")
    return out


INPUT_KSE_REGIME = DATA_DIR / "kse100_with_regime.csv"
INPUT_EPU = DATA_DIR / "pakistan_epu.csv"
INPUT_EXTERNAL = DATA_DIR / "external_features.csv"
OUTPUT_MERGED = DATA_DIR / "kse100_lstm_merged.csv"

SPLIT = (0.70, 0.15, 0.15)     # chronological train / val / test
PURGE_BOUNDARIES = True        # last h rows before each boundary -> 'purge' (context only)
DROP_VOLUME = True
DROP_REDUNDANT = True          # drop `Regime` string (one-hots kept) and `Change` (= Return*100)

# =============================================================================
# 1. Trading calendar: price + indicators + regime
# =============================================================================
kse = pd.read_csv(INPUT_KSE_REGIME)
kse["Date"] = pd.to_datetime(kse["Date"])
kse = kse.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
# indicators were computed on full history upstream, so cutting here loses no warm-up
kse = kse[kse["Date"] >= START_DATE].reset_index(drop=True)
print(f"KSE-100 from {kse['Date'].min().date()} to {kse['Date'].max().date()}  ({len(kse)} trading days)")

if DROP_VOLUME and "Volume" in kse.columns:
    n_missing = kse["Volume"].isna().sum()
    print(f"Dropping Volume: {n_missing}/{len(kse)} rows missing ({n_missing/len(kse):.0%})")
    kse = kse.drop(columns=["Volume"])

# =============================================================================
# 2. EPU (official + Dawn) -- daily, already publication-lagged upstream
# =============================================================================
epu = pd.read_csv(INPUT_EPU)
epu["Date"] = pd.to_datetime(epu["Date"])
epu = epu.drop_duplicates(subset="Date").sort_values("Date")
epu_cols = [c for c in epu.columns if c != "Date"]
merged = kse.merge(epu, on="Date", how="left")
print(f"EPU columns: {epu_cols}")

# =============================================================================
# 3. External market / macro features (already 1-day lagged upstream)
# =============================================================================
ext_cols = []
if INPUT_EXTERNAL.exists():
    ext = pd.read_csv(INPUT_EXTERNAL)
    ext["Date"] = pd.to_datetime(ext["Date"])
    ext = ext.drop_duplicates("Date").sort_values("Date")
    ext_cols = [c for c in ext.columns if c != "Date"]
    merged = merged.merge(ext, on="Date", how="left")
    # trading day with no usable (non-stale) value -> no change; days-since-decision keeps NaN check below
    fill0 = [c for c in ext_cols if c != "X_DAYS_POLICY"]
    merged[fill0] = merged[fill0].fillna(0.0)
    print(f"External columns ({len(ext_cols)}): {ext_cols}")
else:
    print(f"[skip] {INPUT_EXTERNAL.name} not found -> no external features")

# =============================================================================
# 4. News sentiment -- per calendar day, weekend/holiday news rolled to next trading day
# =============================================================================
frames = []
for f in NEWS_FILES:
    p = DATA_DIR / f
    if not p.exists():
        print(f"[skip] news file not found: {f}")
        continue
    frames.append(pd.read_csv(p))
sentiment_cols = ["sentiment_mean", "sentiment_sum", "article_count",
                  "positive_count", "negative_count", "neutral_count", "confidence_mean"]

if frames:
    news = pd.concat(frames, ignore_index=True)
    news["date"] = parse_dates(news["date"])
    bad = news["date"].isna()
    future = news["date"] > min(pd.Timestamp.today().normalize(), kse["Date"].max())
    print(f"News dates: {int(bad.sum())} unparseable, {int(future.sum())} after the last trading day "
          f"({kse['Date'].max().date()}) -> dropped")
    news = news[~bad & ~future]
    key = [c for c in ("url", "link") if c in news.columns][:1] or \
          [c for c in ("date", "title") if c in news.columns]
    n0 = len(news)
    news = news.drop_duplicates(subset=key).reset_index(drop=True)
    print(f"News: {n0} -> {len(news)} articles after de-duplication on {key}")
    news = news[news["date"] >= START_DATE]
    print(f"News in window: {len(news)} articles, {news['date'].min().date()} -> {news['date'].max().date()}")
    if news["date"].min() > START_DATE + pd.Timedelta(days=30):
        print(f"  WARNING: news starts {news['date'].min().date()}, after {START_DATE.date()}. "
              f"Earlier days get sentiment 0 / has_news 0. Pass the 2013+ labeled file via FYP_NEWS_FILES.")

    news["sentiment_score"] = news["prob_positive"] - news["prob_negative"]
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

    # ---- relevance-filtered article-level aggregation (new features) ----
    txt = pd.Series("", index=news.index)
    text_cols = [c for c in ("title", "raw_text", "text") if c in news.columns]
    for c in text_cols:
        txt = txt + " " + news[c].fillna("").astype(str)
    txt = txt.str.lower()
    if USE_RELEVANCE and text_cols:
        news["_rel"] = txt.str.contains("|".join(RELEVANCE_PATTERNS), regex=True)
        print(f"Relevance filter: {news['_rel'].mean():.1%} of articles kept ({int(news['_rel'].sum())}/{len(news)})")
    else:
        news["_rel"] = True
        if USE_RELEVANCE:
            print("  WARNING: no title/raw_text/text column in news file -> relevance filter skipped")
    news["_negflag"] = (lbl == "negative")
    td = merged["Date"].sort_values().to_numpy()
    pos = np.searchsorted(td, news["date"].values, side="left")
    ok = (pos < len(td)) & news["_rel"].values
    r = news.loc[ok].copy()
    r["Date"] = td[pos[ok]]
    rel_agg = r.groupby("Date").agg(rel_sent_mean=("sentiment_score", "mean"),
                                    rel_count=("sentiment_score", "count"),
                                    rel_neg=("_negflag", "sum")).reset_index()
    rel_agg["Date"] = pd.to_datetime(rel_agg["Date"])
    merged = merged.merge(rel_agg, on="Date", how="left")

    trading_days = merged["Date"].sort_values().to_numpy()
    all_days = pd.date_range(daily_news["Date"].min(), merged["Date"].max(), freq="D")
    idx = np.searchsorted(trading_days, all_days.values, side="left")
    idx = np.clip(idx, 0, len(trading_days) - 1)
    day_to_trading_day = dict(zip(all_days, trading_days[idx]))
    daily_news["Date"] = daily_news["Date"].map(day_to_trading_day)

    sent_agg = daily_news.groupby("Date").agg(
        sentiment_mean=("sentiment_mean", "mean"),
        sentiment_sum=("sentiment_sum", "sum"),
        article_count=("article_count", "sum"),
        positive_count=("positive_count", "sum"),
        negative_count=("negative_count", "sum"),
        neutral_count=("neutral_count", "sum"),
        confidence_mean=("confidence_mean", "mean"),
    ).reset_index()
    sent_agg["Date"] = pd.to_datetime(sent_agg["Date"])
    merged = merged.merge(sent_agg, on="Date", how="left")
else:
    print("No news files loaded -> sentiment columns will be zero")
    for c in sentiment_cols:
        merged[c] = np.nan
    merged["rel_sent_mean"], merged["rel_count"], merged["rel_neg"] = np.nan, 0, 0

for col in sentiment_cols:
    merged[col] = merged[col].fillna(0)
merged["rel_count"] = merged["rel_count"].fillna(0)
merged["rel_neg"] = merged["rel_neg"].fillna(0)

# =============================================================================
# 5. Drop rows that cannot have complete features
# =============================================================================
required = ["EPU", "RSI_14", "Return", "Roll_Return_20", "Roll_Vol_20", "Regime"]
required += [c for c in epu_cols if c.startswith("DAWN_") or c in ("EPU_chg", "EPU_dev")]
required += [c for c in ext_cols if c == "X_DAYS_POLICY"]
required = [c for c in dict.fromkeys(required) if c in merged.columns]
before = len(merged)
dropped = merged[merged[required].isna().any(axis=1)]
merged = merged.dropna(subset=required).reset_index(drop=True)
print(f"Dropped {before - len(merged)} rows with missing required features ({before} -> {len(merged)})")
if len(dropped):
    print(f"  dropped dates: {dropped['Date'].min().date()} -> {dropped['Date'].max().date()}")
print(f"First usable row: {merged['Date'].min().date()}  (requested start {START_DATE.date()})")

merged.to_csv(OUTPUT_MERGED, index=False)
print(f"Saved merged features: {OUTPUT_MERGED.name} ({merged.shape[0]} x {merged.shape[1]})")

# =============================================================================
# 6. Feature cleanup + rolling sentiment (past-only)
# =============================================================================
feat = merged.copy()

for r in ["Bear", "Bull", "Sideways"]:
    col = f"Regime_{r}"
    if col not in feat.columns:
        feat[col] = (feat["Regime"].str.title() == r)
    feat[col] = feat[col].astype(int)

feat["has_news"] = (feat["article_count"] > 0).astype(int)
feat["sentiment_mean_3d"] = feat["sentiment_mean"].rolling(3, min_periods=1).mean()
feat["sentiment_mean_7d"] = feat["sentiment_mean"].rolling(7, min_periods=1).mean()

# ---- new sentiment features: past-only, computed on contiguous trading rows ----
W = SURPRISE_WINDOW
def _z(x):
    base = x.shift(1)
    m = base.rolling(W, min_periods=5).mean()
    sd = base.rolling(W, min_periods=5).std().replace(0, np.nan)
    return ((x - m) / sd).clip(-5, 5).fillna(0.0)

rs = feat["rel_sent_mean"]                        # NaN on days without relevant news
feat["sent_surprise"] = _z(rs)
for hl in (1, 3, 10):                             # NaN days carry the decayed value forward
    feat[f"sent_ewm_hl{hl}"] = rs.ewm(halflife=hl).mean().fillna(0.0)
feat["neg_share_7d"] = (feat["rel_neg"].rolling(7, min_periods=1).sum() /
                        feat["rel_count"].rolling(7, min_periods=1).sum().replace(0, np.nan)).fillna(0.0)
feat["count_z"] = _z(np.log1p(feat["rel_count"]).where(feat["rel_count"] > 0))
feat = feat.drop(columns=["rel_sent_mean", "rel_count", "rel_neg"])
SENT_COMPACT = ["sent_surprise", "sent_ewm_hl3", "neg_share_7d", "count_z"]

if DROP_REDUNDANT:
    feat = feat.drop(columns=[c for c in ["Regime", "Change"] if c in feat.columns])

# =============================================================================
# 7. Targets + chronological split
# =============================================================================
def add_targets(df, horizons):
    out = df.copy()
    for h in horizons:
        out[f"target_close_h{h}"] = out["Close"].shift(-h)
        out[f"target_logret_h{h}"] = np.log(out[f"target_close_h{h}"] / out["Close"])
        out[f"target_date_h{h}"] = out["Date"].shift(-h)
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

for h in HORIZONS:
    d = add_split(add_targets(feat, [h]), max_h=h)
    path = DATA_DIR / f"kse100_lstm_h{h}.csv"
    d.to_csv(path, index=False)
    report(path.name, d, [h])

multi = add_split(add_targets(feat, HORIZONS), max_h=max(HORIZONS))
path = DATA_DIR / "kse100_lstm_multi.csv"
multi.to_csv(path, index=False)
report(path.name, multi, HORIZONS)

feature_cols = [c for c in multi.columns
                if not c.startswith("target_") and c not in ("Date", "split")]
print(f"\nInput features ({len(feature_cols)}): {feature_cols}")
print(f"Compact sentiment set: {SENT_COMPACT}")
print("\nNotes for training:")
print("  - split=='purge' rows are input context only; sample windows only from train/val/test rows")
print("  - fit scalers on rows where split == 'train' only")
print("  - train on target_logret_h*, convert back with Close[t] * exp(pred)")