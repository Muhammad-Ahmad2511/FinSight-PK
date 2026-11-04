"""
FinSight PK - Step 4 (v2): Pakistan EPU features, no look-ahead.

Changes vs the old cleaner
  1. Window starts 2013-01-01 (matches the news collection).
  2. Official EPU is spliced properly: EPU-2 (from Aug 2010) is rescaled onto the EPU-4 scale using
     the overlap, instead of pasting raw EPU-2 values next to EPU-4 values.
  3. No look-ahead: a month's value becomes visible only PUBLISH_DELAY_DAYS after that month ends
     (old version showed the full-month value on day 1 of the month).
  4. Instead of the level only, adds change and deviation features.
  5. NEW daily Dawn-based EPU computed from your own scraped articles (trailing window), so it moves
     every day and exists for the whole news period.

Output pakistan_epu.csv (daily), columns:
  Date
  EPU            official monthly level, published-lag aligned (spliced EPU-2/EPU-4)
  EPU_chg        log(EPU) - log(EPU one month earlier)         (month-over-month change)
  EPU_dev        log(EPU) - mean(log EPU) of previous 12 published months (deviation from recent norm)
  DAWN_EPU       % of Dawn articles in the trailing DAWN_WINDOW calendar days that mention
                 economy AND policy AND uncertainty terms (BBD/Baker-Bloom-Davis style)
  DAWN_EPU_chg   DAWN_EPU minus its value 30 days earlier

Run locally:  python 04_build_epu.py
      Put the script, pakistan_epu_raw.xlsx and the Dawn news CSV in ONE folder; it reads/writes there.
      Need: pip install pandas numpy openpyxl.   Optional: set FYP_DATA_DIR to use another folder.
Inputs: pakistan_epu_raw.xlsx (policyuncertainty.com Pakistan monthly) and the labeled Dawn news CSV
        (needs a title/raw_text/text column and a date column).
"""
import os
import re
import sys
import glob
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(os.environ.get("FYP_DATA_DIR", Path(__file__).resolve().parent))   # folder of this script

START_DATE = "2013-01-01"
PUBLISH_DELAY_DAYS = 15        # month m is usable from (end of m) + 15 days. 0 = first day of next month.
DAWN_WINDOW = 30               # trailing calendar days for the Dawn EPU
DAWN_MIN_ARTICLES = 20         # fewer articles in the window -> carry the last value forward

# Dawn EPU rule. The scraped text is only the lead paragraph (~280 chars), so the strict BBD rule
# (economy AND policy AND uncertainty) flags ~0.3% of articles = too sparse/noisy for a daily series.
# All articles are already from the business section, so the "economy" term is dropped and the
# uncertainty vocabulary is widened. Flag = POLICY term AND UNCERTAINTY term.  (set STRICT=True for BBD)
STRICT = False
ECON = [r"\becono(?:my|mic|mics|mically)\b"]
POLICY = [r"\bpolic(?:y|ies)\b", r"\bsbp\b", r"\bfbr\b", r"\bgovernment\b", r"\bbudget\b", r"\btax(?:es|ation)?\b",
          r"\bcentral bank\b", r"\bstate bank\b", r"\bregulat\w+", r"\bdeficit\b", r"\bimf\b",
          r"\bfiscal\b", r"\bmonetary\b", r"\btariff\w*\b", r"\bsubsid\w+", r"\blegislat\w+", r"\bparliament\b"]
UNCERT_STRICT = [r"\buncertain\w*", r"\bunpredictab\w+", r"\bunclear\b", r"\bunstable\b", r"\binstability\b", r"\bvolatil\w+"]
UNCERT = UNCERT_STRICT + [r"\brisk\w*", r"\bconcern\w*", r"\bfears?\b", r"\bworr\w+", r"\bturmoil\b", r"\bcrisis\b"]


def parse_month(val):
    s = str(val).strip()
    s_norm = re.sub(r"(?<=\d)M(?=\d)", "-", s.upper())
    for fmt in ("%b-%y", "%B-%y", "%Y-%m", "%b-%Y", "%B-%Y", "%m/%d/%Y", "%Y-%m-%d"):
        try:
            return pd.to_datetime(s_norm, format=fmt)
        except Exception:
            continue
    return pd.to_datetime(val, errors="coerce")


def load_official():
    raw = pd.read_csv("Pakistan_Policy_Uncertainty_Data(Pakistan EPU Indices).csv", header=None)
    
    header_row = 0
    for i in range(min(10, len(raw))):
        if raw.iloc[i].astype(str).str.lower().str.contains("epu|month|year", regex=True).any():
            header_row = i
            break
    df = pd.read_csv("Pakistan_Policy_Uncertainty_Data(Pakistan EPU Indices).csv", skiprows=header_row)
    df.columns = [str(c).strip() for c in df.columns]
    date_col = next((c for c in df.columns if any(k in c.lower() for k in ("month", "year", "date"))), df.columns[0])
    df["Month"] = df[date_col].apply(parse_month)
    df = df.dropna(subset=["Month"]).sort_values("Month").reset_index(drop=True)
    epu_cols = [c for c in df.columns if "epu" in c.lower()]
    c4 = next((c for c in epu_cols if "4" in c), None)
    c2 = next((c for c in epu_cols if "2" in c), None)
    if c4 is None and c2 is None:
        c4 = epu_cols[0] if epu_cols else next(c for c in df.columns if c not in (date_col, "Month"))
    e4 = pd.to_numeric(df[c4], errors="coerce") if c4 else pd.Series(np.nan, index=df.index)
    e2 = pd.to_numeric(df[c2], errors="coerce") if c2 else pd.Series(np.nan, index=df.index)

    both = e4.notna() & e2.notna()
    if both.sum() >= 12:
        scale = e4[both].mean() / e2[both].mean()
        corr = np.corrcoef(e4[both], e2[both])[0, 1]
        print(f"Splice: EPU-2 x {scale:.3f} onto EPU-4 scale over {both.sum()} overlapping months (corr {corr:.2f})")
    else:
        scale = 1.0
        print("Splice: not enough EPU-2/EPU-4 overlap, EPU-2 used unscaled")
    epu = e4.where(e4.notna(), e2 * scale)
    out = pd.DataFrame({"Month": df["Month"].dt.to_period("M").dt.to_timestamp(), "EPU": epu}).dropna()
    print(f"Official monthly EPU: {len(out)} months, {out.Month.min().date()} -> {out.Month.max().date()}")
    return out.reset_index(drop=True)


def official_daily(monthly, dates):
    m = monthly.copy()
    m["logE"] = np.log(m["EPU"])
    m["chg"] = m["logE"].diff()
    m["dev"] = m["logE"] - m["logE"].shift(1).rolling(12, min_periods=6).mean()   # previous 12 months only
    # visible only after the month is over + publication delay
    m["avail"] = m["Month"] + pd.offsets.MonthEnd(0) + pd.Timedelta(days=PUBLISH_DELAY_DAYS)
    m = m.sort_values("avail")
    d = pd.merge_asof(pd.DataFrame({"Date": dates}), m[["avail", "EPU", "chg", "dev"]],
                      left_on="Date", right_on="avail", direction="backward").drop(columns="avail")
    return d.rename(columns={"chg": "EPU_chg", "dev": "EPU_dev"})


def dawn_epu_daily(dates):
    n = pd.read_csv("dawn_business_2013_2026.csv")
    n["date"] = pd.to_datetime(n["date"])
    txt = (n.get("title", "").fillna("").astype(str) + " " +
           n.get("raw_text", n.get("text", "")).fillna("").astype(str)).str.lower()
    rx = lambda pats: re.compile("|".join(pats))
    has = lambda pats: txt.str.contains(rx(pats), regex=True)
    n["flag"] = ((has(ECON) & has(POLICY) & has(UNCERT_STRICT)) if STRICT else (has(POLICY) & has(UNCERT))).astype(int)
    print(f"Articles {len(n)} ({n.date.min().date()} -> {n.date.max().date()}), "
          f"EPU-flagged {n.flag.mean():.1%}")

    daily = n.groupby("date").agg(flag=("flag", "sum"), n=("flag", "size"))
    cal = pd.date_range(min(daily.index.min(), dates.min()), max(daily.index.max(), dates.max()), freq="D")
    daily = daily.reindex(cal, fill_value=0)
    # trailing window ending on each day (includes the day: news up to that date's close is public)
    f = daily["flag"].rolling(DAWN_WINDOW, min_periods=1).sum()
    a = daily["n"].rolling(DAWN_WINDOW, min_periods=1).sum()
    share = (100 * f / a).where(a >= DAWN_MIN_ARTICLES)
    share = share.ffill()
    out = pd.DataFrame({"DAWN_EPU": share, "DAWN_EPU_chg": share - share.shift(30)})
    out = out.reindex(dates).rename_axis("Date").reset_index()
    return out


def main():
    end = pd.Timestamp.today().normalize()
    news = "dawn_business_2013_2026.csv"
    if news:
        end = pd.to_datetime(pd.read_csv(news, usecols=["date"])["date"]).max()
    dates = pd.date_range(START_DATE, end, freq="D")
    print("=" * 60)
    print(f"FinSight PK - EPU features v2   {START_DATE} -> {end.date()}   publish delay {PUBLISH_DELAY_DAYS}d")
    print("=" * 60)

    off = official_daily(load_official(), dates)
    dw = dawn_epu_daily(dates)
    out = off.merge(dw, on="Date", how="left")
    # official EPU not yet published at the very start -> leave NaN (builder drops those rows) rather than back-fill the future
    out.to_csv(DATA_DIR / "pakistan_epu.csv", index=False)
    print(f"\nSaved pakistan_epu.csv  ({len(out)} daily rows)")
    print("NaN per column:", out.isna().sum().to_dict())
    print(out.dropna().tail(3).to_string(index=False))
    c = out.dropna()[["EPU", "DAWN_EPU"]].corr().iloc[0, 1]
    print(f"corr(official EPU, Dawn EPU) = {c:.2f}   (sanity check: should be clearly positive)")


if __name__ == "__main__":
    main()