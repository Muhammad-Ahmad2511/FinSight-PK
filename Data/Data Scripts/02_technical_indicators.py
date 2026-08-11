"""
FinSight PK - Step 2: Technical indicators (RSI, MACD, EMA)
Manual pandas implementation - no pandas_ta dependency (that package
requires Python 3.12+; this works on any pandas version).

Formulas match the standard definitions (Wilder's RSI, Appel's MACD):
  RSI(14)   : Wilder's smoothed relative strength index
  MACD      : EMA(12) - EMA(26), signal = EMA(9) of MACD line
  EMA(12/26): standard exponential moving average

Input:  kse100_data.csv (from 01_kse100_clean.py)
Output: kse100_with_indicators.csv
"""
import pandas as pd

print("=" * 55)
print("FinSight PK - Technical Indicators (RSI/MACD/EMA)")
print("=" * 55)

df = pd.read_csv("kse100_data.csv", parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# ---- RSI (14-day, Wilder's smoothing) ----
delta = df["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

rs = avg_gain / avg_loss
df["RSI_14"] = 100 - (100 / (1 + rs))

# ---- EMA(12) and EMA(26) ----
df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

# ---- MACD = EMA(12) - EMA(26), signal = EMA(9) of MACD line ----
df["MACD"] = df["EMA_12"] - df["EMA_26"]
df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

# first ~26 rows will have NaNs (indicator warm-up period) - expected,
# these get dropped at the final merge step (09) so all files stay aligned
df.to_csv("kse100_with_indicators.csv", index=False)

print(f"Rows: {len(df)}")
print(f"NaN rows (indicator warm-up, will drop at merge): {df['RSI_14'].isna().sum()}")
print(df[["Date", "Close", "RSI_14", "MACD", "MACD_signal", "EMA_12", "EMA_26"]].tail(3).to_string(index=False))
print("Saved: kse100_with_indicators.csv")