"""
FinSight PK - Step 4: Pakistan EPU Index cleaner
Source is MONTHLY by design (see policyuncertainty.com/pakistan_monthly.html -
EPU-4 covers Jan 2015 onward, EPU-2 goes back further). There is no daily
EPU source for Pakistan, so this script's job is: parse the monthly file
robustly, then stretch each month's value across every day in that month
so it lines up with the daily KSE-100/sentiment files.

Output: pakistan_epu.csv, one row per day, START_DATE..END_DATE
"""
import pandas as pd
import os
import re
from datetime import date

START_DATE = "2025-01-01"
END_DATE = date.today().strftime("%Y-%m-%d")

print("=" * 55)
print("FinSight PK - EPU Index Cleaner (monthly source -> daily)")
print(f"Target window: {START_DATE} -> {END_DATE}")
print("=" * 55)

raw_file = None
for f in ["pakistan_epu_raw.xlsx", "pakistan_epu.xlsx", "pakistan_epu_raw.xls"]:
    if os.path.exists(f):
        raw_file = f
        break
if not raw_file:
    print("ERROR: EPU Excel file not found.")
    print("Download from: https://www.policyuncertainty.com/pakistan_monthly.html")
    exit()

print(f"Loading: {raw_file}")

# --- robust monthly date parsing ---
# policyuncertainty.com files vary in header layout across countries/updates,
# so don't hardcode skiprows/column positions - scan for the row that actually
# holds a parseable month column, whatever its exact label/format is
# (seen formats: "2015M1", "Jan-2015", "1/1/2015", a plain Year+Month pair).
raw = pd.read_excel(raw_file, header=None)

header_row = None
for i in range(min(10, len(raw))):
    row_vals = raw.iloc[i].astype(str).str.lower()
    if row_vals.str.contains("epu|month|year", regex=True).any():
        header_row = i
        break
if header_row is None:
    header_row = 0

df = pd.read_excel(raw_file, skiprows=header_row)
df.columns = [str(c).strip() for c in df.columns]
print(f"Detected header row {header_row}, columns: {list(df.columns)}")

# find the date-like column and the EPU value column(s)
date_col = next((c for c in df.columns if "month" in c.lower() or "year" in c.lower() or "date" in c.lower()), df.columns[0])
epu_cols = [c for c in df.columns if "epu" in c.lower()]
if not epu_cols:
    epu_cols = [c for c in df.columns if c != date_col]

def parse_month(val):
    s = str(val).strip()
    # handles "2015M1", "2015-01", "Jan-2015", "Jan-25" (2-digit year, the
    # format policyuncertainty.com's Pakistan file actually uses), "1/1/2015",
    # Excel datetimes, etc. 2-digit-year formats are tried BEFORE the generic
    # dateutil fallback, since "Jan-25" is ambiguous (dateutil can misread the
    # "25" as a day instead of a year).
    # only collapse a "YYYYMdd"-style M into a dash when it sits BETWEEN
    # digits (e.g. "2015M1" -> "2015-1"). A blanket .replace("M","-") would
    # also corrupt "Mar-25" / "May-25" since those abbreviations contain M.
    s_norm = re.sub(r"(?<=\d)M(?=\d)", "-", s.upper())
    for fmt in ("%b-%y", "%B-%y", "%Y-%m", "%b-%Y", "%B-%Y", "%m/%d/%Y", "%Y-%m-%d"):
        try:
            return pd.to_datetime(s_norm, format=fmt)
        except Exception:
            continue
    return pd.to_datetime(val, errors="coerce")

df["Month"] = df[date_col].apply(parse_month)
df = df.dropna(subset=["Month"]).sort_values("Month").reset_index(drop=True)

# prefer EPU-4 (more newspapers = more coverage) if both variants present,
# fall back to EPU-2 where EPU-4 is missing (EPU-4 only starts Jan 2015)
epu4_col = next((c for c in epu_cols if "4" in c), None)
epu2_col = next((c for c in epu_cols if "2" in c), None)

if epu4_col and epu2_col:
    df["EPU"] = pd.to_numeric(df[epu4_col], errors="coerce").fillna(
        pd.to_numeric(df[epu2_col], errors="coerce"))
elif epu_cols:
    df["EPU"] = pd.to_numeric(df[epu_cols[0]], errors="coerce")
else:
    print("ERROR: could not identify an EPU value column.")
    exit()

df = df.dropna(subset=["EPU"]).reset_index(drop=True)
print(f"Parsed monthly rows: {len(df)}  ({df['Month'].min().date()} -> {df['Month'].max().date()})")

# --- stretch monthly -> daily over the target window ---
daily_dates = pd.date_range(start=START_DATE, end=END_DATE, freq="D")
df_daily = pd.DataFrame({"Date": daily_dates})

df_daily = pd.merge_asof(
    df_daily.sort_values("Date"),
    df[["Month", "EPU"]].rename(columns={"Month": "Date"}).sort_values("Date"),
    on="Date", direction="backward"
)

# if the target window starts before the EPU series' first data point,
# backward-merge leaves leading NaNs - fill those from the earliest known value
n_missing_start = df_daily["EPU"].isna().sum()
if n_missing_start:
    print(f"Backfilling {n_missing_start} leading day(s) with earliest available EPU value")
    df_daily["EPU"] = df_daily["EPU"].bfill()

df_daily.to_csv("pakistan_epu.csv", index=False)

print(f"\nSaved: pakistan_epu.csv")
print(f"Total daily rows: {len(df_daily)}")
print(f"EPU NaN: {df_daily['EPU'].isna().sum()}")
print(df_daily.tail(3).to_string(index=False))
print("Done!")
