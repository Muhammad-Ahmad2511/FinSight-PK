# FinSight PK 📈

**Regime-Aware, Sentiment-Augmented KSE-100 Forecasting with Explainability & Web Deployment**

FAST-NUCES Lahore | Semester 7 & 8 | Fall 2026 | Team of 3

## Problem

Pakistani stock market participants have almost no AI-assisted forecasting tools. Two gaps drive this project:
- No labeled Pakistani financial news sentiment dataset exists.
- KSE-100 forecasting models ignore market regime (bull/bear/sideways) context entirely.

## Approach

FinSight PK is an end-to-end pipeline with four stages:

1. **Sentiment Model** — Headlines scraped day-by-day from Dawn Business (Sep 2014–present), manually annotated, used to fine-tune DistilBERT for daily sentiment scoring.
2. **Technical Indicators** — RSI (14-day, Wilder's smoothing), MACD (EMA-12 − EMA-26, with 9-day signal line), and EMA-12/26 computed via a manual pandas implementation.
3. **Regime Detection** — 20-day rolling return and rolling volatility are computed from KSE-100 daily closes, standardized, and clustered with K-Means (k=3). Clusters are mapped to Bear / Sideways / Bull by ranking their mean rolling return (lowest → Bear, highest → Bull), validated with a silhouette score, and one-hot encoded for downstream use.
4. **Hybrid LSTM Forecasting** — Combines price, technical indicators (RSI, MACD, EMA), sentiment score, regime label, and Pakistan EPU Index to forecast prices 7/14/30 days ahead.
5. **Explainability (SHAP)** — Per-regime SHAP analysis (via `DeepExplainer`) showing which features drive predictions differently across market conditions.

## What Makes It Different

- First KSE-100 model with regime-conditioned forecasting
- First Pakistan-specific financial sentiment dataset/model
- Per-regime (not global) SHAP explainability
- Combines EPU + sentiment + regime in a single pipeline
- Deployed as a public web app, not just a research artifact

## Web Application

- Live KSE-100 dashboard (price, volatility, regime)
- Multi-horizon forecasts with confidence intervals
- Daily sentiment feed
- SHAP explainability panel
- REST API for third-party integration

## Tech Stack

DistilBERT (HuggingFace) · scikit-learn (K-Means) · LSTM · SHAP · pandas-ta · Yahoo Finance/PSX data

## Status

🚧 In development — Fall 2026
