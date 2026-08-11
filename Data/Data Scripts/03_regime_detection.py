"""
FinSight PK - Step 3: Market regime labels (Bull / Bear / Sideways)
K-Means (k=3) on (20-day rolling return, 20-day rolling volatility).
This is the proposal's headline novel component (Fin.pdf Sec 3.2).

Input:  kse100_with_indicators.csv
Output: kse100_with_regime.csv (adds Regime + one-hot columns)
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

print("=" * 55)
print("FinSight PK - Regime Detection (K-Means)")
print("=" * 55)

df = pd.read_csv("kse100_with_indicators.csv", parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

df["Return"] = df["Close"].pct_change()
df["Roll_Return_20"] = df["Return"].rolling(20).mean()
df["Roll_Vol_20"] = df["Return"].rolling(20).std()

features = df[["Roll_Return_20", "Roll_Vol_20"]].dropna()
X = StandardScaler().fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

sil = silhouette_score(X, labels)
print(f"Silhouette score: {sil:.3f}  (>0.5 good, 0.25-0.5 acceptable, <0.25 weak)")

df.loc[features.index, "cluster"] = labels

# map cluster id -> Bull/Bear/Sideways by mean return per cluster
cluster_means = df.groupby("cluster")["Roll_Return_20"].mean().sort_values()
regime_map = {
    cluster_means.index[0]: "Bear",
    cluster_means.index[1]: "Sideways",
    cluster_means.index[2]: "Bull",
}
df["Regime"] = df["cluster"].map(regime_map)
df = df.drop(columns=["cluster"])

# one-hot encode for LSTM input
regime_dummies = pd.get_dummies(df["Regime"], prefix="Regime")
df = pd.concat([df, regime_dummies], axis=1)

df.to_csv("kse100_with_regime.csv", index=False)

print(f"\nRegime distribution:\n{df['Regime'].value_counts()}")
print("Saved: kse100_with_regime.csv")
