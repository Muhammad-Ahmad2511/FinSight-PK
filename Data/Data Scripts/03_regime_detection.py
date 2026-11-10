"""
FinSight PK - Step 3 (v2): Market regime labels (Bull / Bear / Sideways), fitted WITHOUT test-period data.
K-Means (k=3) on (20-day rolling return, 20-day rolling volatility), both trailing (past-only).

Change vs v1: the scaler and K-Means centroids are fitted only on rows up to FIT_END (the end of the
training period). Later rows are assigned to the nearest fitted centroid (predict), so validation/test
days no longer influence where the regime boundaries lie. Cluster -> Bull/Bear/Sideways mapping uses the
mean 20-day return of each cluster in the fit sample.

Input:  kse100_with_indicators.csv
Output: kse100_with_regime.csv (adds Regime + one-hot columns)
Env:    FYP_DATA_DIR (default: script folder), FYP_REGIME_FIT_END (default 2022-08-30; empty = fit on all rows, old behaviour)
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

DATA_DIR = Path(os.environ.get("FYP_DATA_DIR", Path(__file__).resolve().parent))
FIT_END = os.environ.get("FYP_REGIME_FIT_END", "2022-08-30").strip()

print("=" * 55)
print("FinSight PK - Regime Detection (K-Means, train-period fit)")
print("=" * 55)

df = pd.read_csv(DATA_DIR / "kse100_with_indicators.csv", parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

df["Return"] = df["Close"].pct_change()
df["Roll_Return_20"] = df["Return"].rolling(20).mean()
df["Roll_Vol_20"] = df["Return"].rolling(20).std()

features = df[["Roll_Return_20", "Roll_Vol_20"]].dropna()
fit_mask = (df.loc[features.index, "Date"] <= pd.Timestamp(FIT_END)).values if FIT_END else np.ones(len(features), bool)
print(f"Fitting scaler + K-Means on {int(fit_mask.sum())} rows"
      + (f" up to {FIT_END}" if FIT_END else " (all rows)")
      + f"; assigning {int((~fit_mask).sum())} later rows to the nearest centroid")

scaler = StandardScaler().fit(features[fit_mask])
Xfit = scaler.transform(features[fit_mask])
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(Xfit)
labels = kmeans.predict(scaler.transform(features))

sil = silhouette_score(Xfit, kmeans.labels_)
print(f"Silhouette score (fit sample): {sil:.3f}  (>0.5 good, 0.25-0.5 acceptable, <0.25 weak)")

df.loc[features.index, "cluster"] = labels

# map cluster id -> Bull/Bear/Sideways by mean 20-day return per cluster, FIT sample only
fit_idx = features.index[fit_mask]
cluster_means = df.loc[fit_idx].groupby("cluster")["Roll_Return_20"].mean().sort_values()
regime_map = {cluster_means.index[0]: "Bear", cluster_means.index[1]: "Sideways", cluster_means.index[2]: "Bull"}
df["Regime"] = df["cluster"].map(regime_map)
df = df.drop(columns=["cluster"])

regime_dummies = pd.get_dummies(df["Regime"], prefix="Regime")
df = pd.concat([df, regime_dummies], axis=1)
df.to_csv(DATA_DIR / "kse100_with_regime.csv", index=False)

print(f"\nRegime distribution (all rows):\n{df['Regime'].value_counts()}")
if FIT_END:
    late = df[df["Date"] > pd.Timestamp(FIT_END)]
    print(f"\nRegime distribution after {FIT_END}:\n{late['Regime'].value_counts()}")
print("Saved: kse100_with_regime.csv")