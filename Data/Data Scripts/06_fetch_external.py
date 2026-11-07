"""
FinSight PK - Step 5: external market / macro features (free, no API key).

Global (Yahoo Finance via yfinance):  S&P 500, emerging-markets ETF, oil, gold, US dollar index, VIX
Rupee (Yahoo):                        USD/PKR daily and 20-day change
Rates (your CSV):                     SBP policy-rate change at the last decision + days since that decision

Rules that keep it leakage-free
  * only CHANGES (log returns / differences) are used, never levels
  * every value is shifted by 1 calendar day: a close dated d is first usable on d+1.
    (PSX closes ~15:30 PKT, before the US open, so a US close dated t is future information on PSX day t.)
  * weekends/holidays: the last available value is carried forward for at most MAX_STALE_DAYS, then NaN

Output: external_features.csv   (one row per calendar day, columns Date, X_*)
Run:    pip install yfinance pandas numpy      then      python 05_fetch_external.py
        (works on Colab or locally; reads/writes the script folder, or FYP_DATA_DIR)

Policy-rate file  sbp_policy_rate.csv  (YOU fill it, ~20-30 rows, from the SBP monetary policy statements):
    date,rate
    2012-10-08,10.0
    ...
  `date` = announcement date, `rate` = policy rate in % after that decision. Include the decision that was
  in force on START_DATE (i.e. one or two rows BEFORE 2013) and every decision after it - unchanged decisions too.
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(os.environ.get("FYP_DATA_DIR", Path(__file__).resolve().parent))
START_DATE = "2013-01-01"       # first row written
FETCH_FROM = "2012-09-01"       # extra history so the first rows have a previous close / 20-day change
MAX_STALE_DAYS = 10
POLICY_FILE = "sbp_policy_rate.csv"

# name -> Yahoo ticker. Tickers are the usual Yahoo ones; the script reports any that return nothing.
GLOBAL = {"SPX": "^GSPC", "EEM": "EEM", "OIL": "CL=F", "GOLD": "GC=F", "DXY": "DX-Y.NYB", "VIX": "^VIX"}
PKR = "PKR=X"


def download_close(ticker):
    import yfinance as yf
    df = yf.download(ticker, start=FETCH_FROM, auto_adjust=True, progress=False, threads=False)
    if df is None or len(df) == 0:
        return None
    close = df["Close"]
    if isinstance(close, pd.DataFrame):          # newer yfinance returns (field, ticker) columns
        close = close.iloc[:, 0]
    close = close.dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close = close[close > 0]
    return close[~close.index.duplicated(keep="last")].sort_index()


def lagged_daily(s, cal):
    """s indexed by the date the value is known at the close of that date -> usable from the NEXT day."""
    s = s.dropna()
    a = pd.DataFrame({"avail": s.index + pd.Timedelta(days=1), "v": s.values}).sort_values("avail")
    out = pd.merge_asof(pd.DataFrame({"Date": cal}), a, left_on="Date", right_on="avail",
                        direction="backward", tolerance=pd.Timedelta(days=MAX_STALE_DAYS))
    return out["v"].values


def policy_features(cal):
    path = DATA_DIR / POLICY_FILE
    if not path.exists():
        print(f"  [skip] {POLICY_FILE} not found -> no policy-rate features. Create it (see header of this script).")
        return {}
    p = pd.read_csv(path)
    p.columns = [c.strip().lower() for c in p.columns]
    p["date"] = pd.to_datetime(p["date"])
    p = p.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    p["chg"] = p["rate"].astype(float).diff()
    if p["date"].min() > pd.Timestamp(START_DATE):
        print(f"  WARNING: first policy row is {p['date'].min().date()}, after {START_DATE}. "
              "Add the decision in force before that date, otherwise the early rows are filled with 0.")
    p["avail"] = p["date"] + pd.Timedelta(days=1)            # announced during the day -> usable next day
    m = pd.merge_asof(pd.DataFrame({"Date": cal}), p[["avail", "chg"]], left_on="Date", right_on="avail",
                      direction="backward")
    days = (m["Date"] - m["avail"]).dt.days + 1              # days since the decision itself
    print(f"  policy rate: {len(p)} decisions, {p['date'].min().date()} -> {p['date'].max().date()}, "
          f"last rate {p['rate'].iloc[-1]}%")
    return {"X_POLICY_CHG": m["chg"].values, "X_DAYS_POLICY": np.log1p(days.values.astype(float))}


def main(download=download_close):
    end = pd.Timestamp.today().normalize()
    cal = pd.date_range(START_DATE, end, freq="D")
    cols = {}
    print("=" * 60)
    print(f"FinSight PK - external features   {START_DATE} -> {end.date()}   lag 1 day")
    print("=" * 60)

    for name, tk in GLOBAL.items():
        c = download(tk)
        if c is None or len(c) < 100:
            print(f"  [FAIL] {name} ({tk}): no data - check the ticker or your connection; column skipped")
            continue
        r = np.log(c).diff().dropna()
        cols[f"X_{name}"] = lagged_daily(r, cal)
        print(f"  {name:5s} {tk:9s} {len(c):5d} closes  {c.index.min().date()} -> {c.index.max().date()}")

    c = download(PKR)
    if c is None or len(c) < 100:
        print(f"  [FAIL] PKR ({PKR}): no data; column skipped (alternative: SBP interbank USD/PKR csv)")
    else:
        lc = np.log(c)
        cols["X_PKR_1D"] = lagged_daily(lc.diff().dropna(), cal)
        cols["X_PKR_20D"] = lagged_daily(lc.diff(20).dropna(), cal)
        gaps = (c.index.to_series().diff().dt.days > 5).sum()
        print(f"  PKR   {PKR:9s} {len(c):5d} closes  {c.index.min().date()} -> {c.index.max().date()}  (gaps >5 days: {gaps})")

    cols.update(policy_features(cal))
    if not cols:
        sys.exit("Nothing produced - no ticker downloaded and no policy file.")

    out = pd.DataFrame({"Date": cal, **cols})
    out.to_csv(DATA_DIR / "external_features.csv", index=False)
    print(f"\nSaved external_features.csv  ({len(out)} rows, {len(cols)} features)")
    print("NaN share per feature:", {k: f"{v:.1%}" for k, v in out.drop(columns="Date").isna().mean().items()})
    print(out.dropna().tail(3).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
