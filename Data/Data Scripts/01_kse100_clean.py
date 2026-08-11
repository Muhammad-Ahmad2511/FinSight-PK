"""
FinSight PK - Step 1: KSE-100 OHLCV cleaner
Source: Investing.com manual export -> "Karachi 100 Historical Data.csv"
Output: kse100_data.csv, restricted to START_DATE..END_DATE so every
        downstream dataset (EPU, sentiment, indicators, regime) shares
        the exact same date window.
"""
import pandas as pd
import os
from datetime import date

START_DATE = "2025-01-01"
END_DATE = date.today().strftime("%Y-%m-%d")  # today, dynamic

RAW_FILE = "Karachi 100 Historical Data.csv"

print("=" * 55)
print("FinSight PK - KSE-100 Data Cleaner")
print(f"Target window: {START_DATE} -> {END_DATE}")
print("=" * 55)

if not os.path.exists(RAW_FILE):
    print(f"ERROR: '{RAW_FILE}' not found. Download it from investing.com first.")
    exit()

df = pd.read_csv(RAW_FILE)
print(f"Raw file loaded: {len(df)} rows")

col_map = {}
for col in df.columns:
    c = col.strip().lower()
    if "date" in c: col_map[col] = "Date"
    elif "open" in c: col_map[col] = "Open"
    elif "high" in c: col_map[col] = "High"
    elif "low" in c: col_map[col] = "Low"
    elif "close" in c or "price" in c: col_map[col] = "Close"
    elif "vol" in c: col_map[col] = "Volume"
    elif "change" in c: col_map[col] = "Change"
df = df.rename(columns=col_map)

df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
df = df.dropna(subset=["Date"])

# --- hard date filter: everything else in the pipeline uses this same window ---
df = df[(df["Date"] >= START_DATE) & (df["Date"] <= END_DATE)]
df = df.sort_values("Date").reset_index(drop=True)

def parse_volume(v):
    v = str(v).replace(",", "").strip()
    if v in ["nan", "", "-"]:
        return None
    suffixes = {"K": 1e3, "M": 1e6, "B": 1e9}
    if v[-1].upper() in suffixes:
        try:
            return float(v[:-1]) * suffixes[v[-1].upper()]
        except Exception:
            return None
    try:
        return float(v)
    except Exception:
        return None

for col in ["Open", "High", "Low", "Close"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace(",", "").str.strip()
        df[col] = pd.to_numeric(df[col], errors="coerce")

if "Volume" in df.columns:
    df["Volume"] = df["Volume"].apply(parse_volume)

if "Change" in df.columns:
    df["Change"] = df["Change"].astype(str).str.replace("%", "").str.strip()
    df["Change"] = pd.to_numeric(df["Change"], errors="coerce")

df = df.dropna(subset=["Close"]).reset_index(drop=True)

df.to_csv("kse100_data.csv", index=False)

print(f"\nClean file ready: kse100_data.csv")
print(f"Total trading days: {len(df)}")
print(f"Date range: {df['Date'].min().date()} -> {df['Date'].max().date()}")
print(f"Volume NaN count: {df['Volume'].isna().sum() if 'Volume' in df.columns else 'N/A'}")
print(df.tail(3).to_string(index=False))
print("Done!")
