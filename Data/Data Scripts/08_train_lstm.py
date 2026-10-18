"""
Train, tune and evaluate the FinSight multi-horizon LSTM (3 / 7 / 14 trading days).

Input : kse100_lstm_multi.csv and kse100_lstm_features.csv from 07_build_lstm_dataset.py
Output: everything in <data_dir>/lstm_outputs (override with --out)

Pipeline
  1. Feature transform: price levels -> scale-free ratios (Open/High/Low/EMA/MACD relative to Close),
     counts -> log1p, EPU -> log. Raw Close is NOT a feature (it is non-stationary: 27k -> 189k).
  2. Targets: log(Close[t+h]/Close[t]) for h = 3, 7, 14, standardised with train statistics.
  3. Hyper-parameter search: Optuna (TPE + median pruning) over architecture, dropout, learning rate,
     weight decay, batch size, loss, FEATURE GROUPS (EPU on/off, sentiment none/core/all, regime on/off)
     and TARGET SCALING (raw log-return vs log-return divided by recent volatility * sqrt(h)). Objective = validation MSE relative to the train-mean-drift forecast
     (below 1 beats drift), in raw log-return units, averaged over horizons and over EXPANDING-WINDOW time-series
     folds taken from the pre-test period, with a purge gap of max(h) rows between train and validation,
     and scalers refit on each fold's training rows. The test set is never touched during tuning.
  4. Final model: seed ensemble trained on the 'train' split, early-stopped on 'val'.
  5. Uncertainty: Monte Carlo dropout (dropout stays on at inference, M passes per ensemble member,
     pooled). The point forecast is the MC mean. MC spread alone badly under-covers, so the interval
     scale is sqrt(MC variance + validation residual variance) times a multiplier k calibrated on the
     validation split (normalised-residual conformal quantile). Raw and calibrated test coverage are
     both reported.
  6. Evaluation on the held-out test split vs baselines (random walk, train-drift, Ridge) and an
     ablation (price-only, +EPU, +sentiment, +regime, no-regime, no-sentiment, full) with
     Diebold-Mariano tests, per-regime errors and interval coverage.
  7. Saves the final model + a latest 3/7/14-day forecast from the last rows of the features file.

Requirements: torch numpy pandas scikit-learn optuna matplotlib
Usage:
  python 08_train_lstm.py                         # tune + train + evaluate (default 40 trials)
  python 08_train_lstm.py --trials 80 --seeds 5
  python 08_train_lstm.py --params lstm_outputs/best_params.json   # skip tuning
  python 08_train_lstm.py --quick                 # tiny smoke test
Env var FYP_DATA_DIR overrides the data folder.
"""
import argparse
import copy
import json
import math
import os
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HORIZONS = (3, 7, 14)
H = len(HORIZONS)
MAX_H = max(HORIZONS)
CLIP = 5.0

# ----------------------------------------------------------------------------------------------
# Feature groups (ablation units)
# ----------------------------------------------------------------------------------------------
GROUPS = {
    "tech": ["open_rel", "high_rel", "low_rel", "ema12_rel", "ema26_rel", "macd_rel", "macd_sig_rel",
             "macd_hist_rel", "rsi", "ret", "roll_ret_20", "roll_vol_20"],
    "epu": ["log_epu"],
    "sentiment": ["sentiment_mean", "sentiment_sum", "log_articles", "log_pos", "log_neg", "log_neu",
                  "confidence_mean", "has_news", "sent_3d", "sent_7d"],
    "regime": ["regime_bear", "regime_bull", "regime_sideways"],
    "sent_core": ["sentiment_mean", "log_articles", "sent_7d"],
}
BINARY = {"has_news", "regime_bear", "regime_bull", "regime_sideways"}   # not standardised
CONFIGS = {
    "price_only": ["tech"],
    "price+epu": ["tech", "epu"],
    "price+sentiment": ["tech", "sentiment"],
    "price+regime": ["tech", "regime"],
    "no_regime": ["tech", "epu", "sentiment"],
    "no_sentiment": ["tech", "epu", "regime"],
    "full": ["tech", "epu", "sentiment", "regime"],
}


def make_features(df):
    """Scale-free model features from the columns written by 07_build_lstm_dataset.py."""
    c = df["Close"]
    f = pd.DataFrame(index=df.index)
    f["open_rel"] = np.log(df["Open"] / c)
    f["high_rel"] = np.log(df["High"] / c)
    f["low_rel"] = np.log(df["Low"] / c)
    f["ema12_rel"] = np.log(df["EMA_12"] / c)
    f["ema26_rel"] = np.log(df["EMA_26"] / c)
    f["macd_rel"] = df["MACD"] / c
    f["macd_sig_rel"] = df["MACD_signal"] / c
    f["macd_hist_rel"] = df["MACD_hist"] / c
    f["rsi"] = df["RSI_14"]
    f["ret"] = df["Return"]
    f["roll_ret_20"] = df["Roll_Return_20"]
    f["roll_vol_20"] = df["Roll_Vol_20"]
    f["log_epu"] = np.log(df["EPU"])
    f["sentiment_mean"] = df["sentiment_mean"]
    f["sentiment_sum"] = df["sentiment_sum"]
    f["log_articles"] = np.log1p(df["article_count"])
    f["log_pos"] = np.log1p(df["positive_count"])
    f["log_neg"] = np.log1p(df["negative_count"])
    f["log_neu"] = np.log1p(df["neutral_count"])
    f["confidence_mean"] = df["confidence_mean"]
    f["has_news"] = df["has_news"].astype(float)
    f["sent_3d"] = df["sentiment_mean_3d"]
    f["sent_7d"] = df["sentiment_mean_7d"]
    f["regime_bear"] = df["Regime_Bear"].astype(float)
    f["regime_bull"] = df["Regime_Bull"].astype(float)
    f["regime_sideways"] = df["Regime_Sideways"].astype(float)
    return f


def cols_from_params(params):
    g = ["tech"]
    if params.get("feat_epu", 1):
        g.append("epu")
    g += {0: [], 1: ["sent_core"], 2: ["sentiment"]}[params.get("feat_sent", 2)]
    if params.get("feat_regime", 1):
        g.append("regime")
    return [c for x in g for c in GROUPS[x]]


def cols_for(config, params=None):
    if config == "tuned":
        return cols_from_params(params)
    return [c for g in CONFIGS[config] for c in GROUPS[g]]


# ----------------------------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------------------------
def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def norm_ppf(p):
    from statistics import NormalDist
    return NormalDist().inv_cdf(p)


# ----------------------------------------------------------------------------------------------
# Data container
# ----------------------------------------------------------------------------------------------
class Data:
    def __init__(self, multi_path):
        df = pd.read_csv(multi_path, parse_dates=["Date"] + [f"target_date_h{h}" for h in HORIZONS])
        df = df.sort_values("Date").reset_index(drop=True)
        # windows are built over consecutive rows, so the file must be contiguous
        for h in HORIZONS:
            ok = (df["Date"].shift(-h).iloc[:-h].values == df[f"target_date_h{h}"].iloc[:-h].values).all()
            assert ok, f"rows are not contiguous (target_date_h{h} mismatch): purge rows must be kept"
        self.df = df
        self.feats = make_features(df)
        assert not self.feats.isna().any().any(), "NaNs in features"
        self.Y = df[[f"target_logret_h{h}" for h in HORIZONS]].values.astype(np.float64)
        self.close = df["Close"].values.astype(np.float64)
        self.vscale = (np.maximum(df["Roll_Vol_20"].values, 1e-3)[:, None]
                       * np.sqrt(np.array(HORIZONS, dtype=float))[None, :])      # past-only volatility scale
        self.ones = np.ones_like(self.vscale)
        self.split = df["split"].values
        self.n = len(df)
        self.idx = np.arange(self.n)

    def vs(self, params):
        return self.vscale if params.get("target_scale", "none") == "vol" else self.ones

    def rows(self, names, window):
        elig = self.idx >= window - 1
        return self.idx[np.isin(self.split, names) & elig]


def scale_features(feats, cols, fit_rows):
    """Standardise continuous columns with statistics from fit_rows only; binary columns pass through."""
    X = feats[cols].values.astype(np.float64)
    mean = np.zeros(len(cols))
    std = np.ones(len(cols))
    for j, c in enumerate(cols):
        if c in BINARY:
            continue
        mean[j] = X[fit_rows, j].mean()
        std[j] = max(X[fit_rows, j].std(), 1e-8)
    Xs = np.clip((X - mean) / std, -CLIP, CLIP).astype(np.float32)
    return Xs, {"cols": cols, "mean": mean, "std": std}


def apply_scaler(feats, scaler):
    X = feats[scaler["cols"]].values.astype(np.float64)
    return np.clip((X - scaler["mean"]) / scaler["std"], -CLIP, CLIP).astype(np.float32)


def all_windows(Xs, window):
    """(n - W + 1, W, F); window j ends at row j + W - 1."""
    w = np.lib.stride_tricks.sliding_window_view(Xs, window, axis=0)   # (n-W+1, F, W)
    return np.ascontiguousarray(w.transpose(0, 2, 1))


def take(Wall, rows, window):
    return Wall[rows - window + 1]


# ----------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------
class LSTMForecaster(nn.Module):
    """Stacked single-layer LSTMs, each followed by an explicit Dropout module so that
    Monte Carlo dropout affects every layer (nn.LSTM's built-in inter-layer dropout is not a module)."""

    def __init__(self, n_feat, hidden, n_layers, dropout, head_dim, n_out=H):
        super().__init__()
        self.lstms = nn.ModuleList()
        self.drops = nn.ModuleList()
        d = n_feat
        for _ in range(n_layers):
            lstm = nn.LSTM(d, hidden, batch_first=True)
            for name, p in lstm.named_parameters():       # forget-gate bias = 1
                if "bias_ih" in name:
                    p.data[hidden:2 * hidden].fill_(1.0)
            self.lstms.append(lstm)
            self.drops.append(nn.Dropout(dropout))
            d = hidden
        self.head = nn.Sequential(nn.Linear(hidden, head_dim), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(head_dim, n_out))

    def forward(self, x):
        for lstm, drop in zip(self.lstms, self.drops):
            x, _ = lstm(x)
            x = drop(x)
        return self.head(x[:, -1])


def eval_mse(model, W, Y, bs=1024):
    model.eval()
    tot = 0.0
    with torch.no_grad():
        for i in range(0, len(W), bs):
            p = model(W[i:i + bs])
            tot += ((p - Y[i:i + bs]) ** 2).sum().item()
    return tot / Y.numel()


def train_model(params, Wtr, Ytr, Wva, Yva, seed, device, max_epochs, patience):
    set_seed(seed)
    model = LSTMForecaster(Wtr.shape[2], params["hidden"], params["n_layers"], params["dropout"],
                           params["head_dim"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3, min_lr=1e-6)
    crit = nn.HuberLoss(delta=1.0) if params["loss"] == "huber" else nn.MSELoss()
    Wtr_t = torch.as_tensor(Wtr, device=device)
    Ytr_t = torch.as_tensor(Ytr, dtype=torch.float32, device=device)
    Wva_t = torch.as_tensor(Wva, device=device)
    Yva_t = torch.as_tensor(Yva, dtype=torch.float32, device=device)
    n, bs = len(Wtr_t), params["batch_size"]
    best, best_state, best_ep, bad, hist = float("inf"), None, 0, 0, []
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        tr_loss = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = crit(model(Wtr_t[idx]), Ytr_t[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * len(idx)
        val = eval_mse(model, Wva_t, Yva_t)
        sched.step(val)
        hist.append((tr_loss / n, val))
        if val < best - 1e-5:
            best, best_ep, bad = val, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best, best_ep, hist


@torch.no_grad()
def point_predict(model, W, device, bs=1024):
    model.eval()
    Wt = torch.as_tensor(W, device=device)
    return torch.cat([model(Wt[i:i + bs]) for i in range(0, len(Wt), bs)]).cpu().numpy()


@torch.no_grad()
def mc_predict(model, W, M, device, bs=1024):
    """Dropout active at inference. Returns (M, N, H) in standardised-target units."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    Wt = torch.as_tensor(W, device=device)
    outs = []
    for _ in range(M):
        outs.append(torch.cat([model(Wt[i:i + bs]) for i in range(0, len(Wt), bs)]).cpu().numpy())
    model.eval()
    return np.stack(outs)


# ----------------------------------------------------------------------------------------------
# Hyper-parameter search
# ----------------------------------------------------------------------------------------------
def suggest(trial, window, tune_window):
    return {
        "hidden": trial.suggest_categorical("hidden", [16, 32, 64, 128]),
        "n_layers": trial.suggest_int("n_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.1, 0.6),
        "head_dim": trial.suggest_categorical("head_dim", [16, 32, 64]),
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "loss": trial.suggest_categorical("loss", ["mse", "huber"]),
        "window": trial.suggest_categorical("window", [20, 30, 60]) if tune_window else window,
        "feat_epu": trial.suggest_int("feat_epu", 0, 1),
        "feat_sent": trial.suggest_int("feat_sent", 0, 2),          # 0 none, 1 core(3), 2 all(10)
        "feat_regime": trial.suggest_int("feat_regime", 0, 1),
        "target_scale": trial.suggest_categorical("target_scale", ["none", "vol"]),
    }


def make_folds(data, window, n_folds, start=0.55):
    """Expanding-window folds on the pre-test samples. Validation blocks are consecutive slices of the
    eligible rows; training rows must end MAX_H rows before the block starts (no target overlap)."""
    E = data.rows(["train", "val"], window)
    edges = np.linspace(start, 1.0, n_folds + 1)
    folds = []
    for k in range(n_folds):
        va = E[int(len(E) * edges[k]):int(len(E) * edges[k + 1])]
        tr = E[E <= va[0] - MAX_H - 1]
        folds.append((tr, va))
    return folds


def cv_loss(params, data, cols, folds, device, max_epochs, patience, seed, trial=None):
    """Mean over folds of  MSE(model) / MSE(train-mean drift)  in raw log-return units, averaged over
    horizons. Comparable across feature sets and target scalings; < 1 means the model beats drift."""
    import optuna
    window = params["window"]
    vs = data.vs(params)
    losses = []
    for k, (tr, va) in enumerate(folds):
        fit_rows = data.idx[data.idx <= tr.max()]
        Xs, _ = scale_features(data.feats, cols, fit_rows)
        Wall = all_windows(Xs, window)
        Ye = data.Y / vs
        ym, ys = Ye[tr].mean(0), Ye[tr].std(0)
        Ytr, Yva = (Ye[tr] - ym) / ys, (Ye[va] - ym) / ys
        Wva = take(Wall, va, window)
        model, _, _, _ = train_model(params, take(Wall, tr, window), Ytr, Wva, Yva,
                                     seed, device, max_epochs, patience)
        p = (point_predict(model, Wva, device) * ys + ym) * vs[va]
        drift = data.Y[tr].mean(0)
        ratio = np.mean(np.mean((p - data.Y[va]) ** 2, 0) / np.mean((drift - data.Y[va]) ** 2, 0))
        losses.append(float(ratio))
        if trial is not None:
            trial.report(float(np.mean(losses)), step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return float(np.mean(losses)), losses


def tune(data, args, device, out):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    folds = make_folds(data, args.window, args.folds)
    for k, (tr, va) in enumerate(folds):
        log(f"fold {k}: train {len(tr)} rows (to {data.df.Date[tr.max()].date()}), "
            f"val {len(va)} rows ({data.df.Date[va.min()].date()} -> {data.df.Date[va.max()].date()})")

    def objective(trial):
        params = suggest(trial, args.window, args.tune_window)
        if params["window"] != args.window:      # folds depend on the window length
            fo = make_folds(data, params["window"], args.folds)
        else:
            fo = folds
        mean_loss, losses = cv_loss(params, data, cols_from_params(params), fo, device, args.tune_epochs, args.tune_patience,
                                    args.seed, trial)
        trial.set_user_attr("fold_losses", losses)
        return mean_loss

    study = optuna.create_study(
        direction="minimize", study_name="finsight_lstm",
        sampler=optuna.samplers.TPESampler(seed=args.seed, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0),
        storage=f"sqlite:///{out / 'optuna_study.db'}", load_if_exists=True)

    def cb(study, trial):
        v = f"{trial.value:.4f}" if trial.value is not None else "pruned"
        log(f"trial {trial.number:3d}  {v}  best {study.best_value:.4f}" if study.best_trial else "")

    done = len([t for t in study.trials if t.state.is_finished()])
    remaining = max(0, args.trials - done)
    log(f"tuning: {remaining} trials to run ({done} already in study)")
    if remaining:
        study.optimize(objective, n_trials=remaining, timeout=args.timeout, callbacks=[cb],
                       gc_after_trial=True)
    best = dict(study.best_trial.params)
    best.setdefault("window", args.window)
    study.trials_dataframe().to_csv(out / "tuning_trials.csv", index=False)
    json.dump(best, open(out / "best_params.json", "w"), indent=2)
    log("best params:", json.dumps(best), f"| CV val MSE/drift-MSE {study.best_value:.4f}",
        "| fold losses", [round(x, 4) for x in study.best_trial.user_attrs["fold_losses"]])
    return best, study


# ----------------------------------------------------------------------------------------------
# Ensemble fit / predict
# ----------------------------------------------------------------------------------------------
MAIN = "tuned"       # model whose feature groups / target scaling were chosen by the search


class Ensemble:
    pass


def fit_ensemble(name, params, data, seeds, device, max_epochs, patience, M):
    cols = cols_for(name, params)
    window = params["window"]
    tr, va, te = (data.rows([s], window) for s in ("train", "val", "test"))
    fit_rows = data.idx[data.split == "train"]
    Xs, scaler = scale_features(data.feats, cols, fit_rows)
    Wall = all_windows(Xs, window)
    vs = data.vs(params)
    Ye = data.Y / vs
    ym, ys = Ye[tr].mean(0), Ye[tr].std(0)
    Ytr, Yva = (Ye[tr] - ym) / ys, (Ye[va] - ym) / ys
    ens = Ensemble()
    ens.vs_all = vs
    ens.name, ens.params, ens.cols, ens.scaler, ens.window = name, params, cols, scaler, window
    ens.ym, ens.ys, ens.Wall, ens.device = ym, ys, Wall, device
    ens.M = M if name == MAIN else min(M, 30)     # ablation members only need the point forecast
    ens.models, ens.best_eps, ens.val_losses, ens.hists = [], [], [], []
    for s in seeds:
        m, best, ep, hist = train_model(params, take(Wall, tr, window), Ytr, take(Wall, va, window), Yva,
                                        s, device, max_epochs, patience)
        ens.models.append(m)
        ens.best_eps.append(ep)
        ens.val_losses.append(best)
        ens.hists.append(hist)
    log(f"[{name}] {len(seeds)} seeds | val MSE(std) {np.mean(ens.val_losses):.4f} "
        f"+/- {np.std(ens.val_losses):.4f} | best epochs {ens.best_eps}")
    return ens


def predict(ens, rows):
    """Pooled MC samples in log-return units: (S*M, N, H)."""
    W = take(ens.Wall, rows, ens.window)
    S = np.concatenate([mc_predict(m, W, ens.M, ens.device) for m in ens.models])
    return (S * ens.ys + ens.ym) * ens.vs_all[rows][None]


def conformal_k(mu, sd, y, conf):
    r = np.abs(y - mu) / np.maximum(sd, 1e-9)
    n = len(r)
    level = min(1.0, math.ceil((n + 1) * conf) / n)
    return np.quantile(r, level, axis=0)          # (H,)


# ----------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------
def horizon_metrics(model_name, close_t, Y, mu):
    rows = []
    for j, h in enumerate(HORIZONS):
        y, p = Y[:, j], mu[:, j]
        yc, pc = close_t * np.exp(y), close_t * np.exp(p)
        nz = y != 0
        mse_lr = np.mean((p - y) ** 2)
        rows.append({
            "model": model_name, "horizon": h,
            "RMSE": np.sqrt(np.mean((pc - yc) ** 2)), "MAE": np.mean(np.abs(pc - yc)),
            "MAPE": 100 * np.mean(np.abs(pc - yc) / yc),
            "DirAcc": np.mean(np.sign(p[nz]) == np.sign(y[nz])),
            "logret_RMSE": np.sqrt(mse_lr),
            "skill_vs_RW": 1 - mse_lr / np.mean(y ** 2),
        })
    return rows


def dm_test(e1, e2, h):
    """Diebold-Mariano on squared errors with a Bartlett/Newey-West variance (lag h-1).
    stat < 0 => model 1 has lower loss than model 2."""
    d = e1 ** 2 - e2 ** 2
    T = len(d)
    dbar = d.mean()

    def g(k):
        return np.mean((d[k:] - dbar) * (d[:T - k] - dbar))

    var = g(0) + 2 * sum((1 - k / h) * g(k) for k in range(1, h))
    if var <= 0:
        return float("nan"), float("nan")
    stat = dbar / math.sqrt(var / T)
    return stat, math.erfc(abs(stat) / math.sqrt(2))


def ridge_baseline(data, window, alpha_grid=np.logspace(-3, 4, 15)):
    from sklearn.linear_model import Ridge
    cols = cols_for("full")
    tr, va, te = (data.rows([s], window) for s in ("train", "val", "test"))
    Xs, _ = scale_features(data.feats, cols, data.idx[data.split == "train"])
    ym, ys = data.Y[tr].mean(0), data.Y[tr].std(0)
    best = (float("inf"), None)
    for a in alpha_grid:
        r = Ridge(alpha=a).fit(Xs[tr], (data.Y[tr] - ym) / ys)
        v = np.mean((r.predict(Xs[va]) - (data.Y[va] - ym) / ys) ** 2)
        if v < best[0]:
            best = (v, a)
    r = Ridge(alpha=best[1]).fit(Xs[tr], (data.Y[tr] - ym) / ys)
    return r.predict(Xs[te]) * ys + ym, best[1]


# ----------------------------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------------------------
def make_plots(out, pred_df, ens_full, metrics_df, study):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        log("matplotlib not available, skipping plots")
        return
    fig, axes = plt.subplots(H, 1, figsize=(11, 3.4 * H), sharex=False)
    for ax, h in zip(np.atleast_1d(axes), HORIZONS):
        d = pred_df.dropna(subset=[f"actual_close_h{h}"])
        x = pd.to_datetime(d[f"target_date_h{h}"])
        ax.plot(x, d[f"actual_close_h{h}"], color="#1f2937", lw=1.2, label="actual")
        ax.plot(x, d[f"pred_close_h{h}"], color="#2563eb", lw=1.2, label="forecast (MC mean)")
        ax.fill_between(x, d[f"lo_h{h}"], d[f"hi_h{h}"], color="#2563eb", alpha=0.18, label="calibrated interval")
        ax.set_title(f"KSE-100, {h}-trading-day forecast (test set)")
        ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "test_forecasts.png", dpi=130)
    plt.close()

    ab = metrics_df[~metrics_df.model.isin(["random_walk"])]
    piv = ab.pivot(index="model", columns="horizon", values="skill_vs_RW")
    ax = piv.plot(kind="bar", figsize=(10, 4.5))
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("skill vs random walk (log-return MSE)")
    ax.set_title("Test skill by model (above 0 beats random walk)")
    plt.tight_layout()
    plt.savefig(out / "ablation_skill.png", dpi=130)
    plt.close()

    plt.figure(figsize=(7, 4))
    for i, h in enumerate(ens_full.hists):
        a = np.array(h)
        plt.plot(a[:, 0], color="#2563eb", alpha=0.4, label="train loss" if i == 0 else None)
        plt.plot(a[:, 1], color="#dc2626", alpha=0.4, label="val MSE" if i == 0 else None)
    plt.xlabel("epoch")
    plt.legend()
    plt.title("Training curves (full model, all seeds)")
    plt.tight_layout()
    plt.savefig(out / "training_curves.png", dpi=130)
    plt.close()

    if study is not None:
        vals = [t.value for t in study.trials if t.value is not None]
        plt.figure(figsize=(7, 4))
        plt.plot(vals, ".", alpha=0.5)
        plt.plot(np.minimum.accumulate(vals), color="#dc2626")
        plt.xlabel("trial")
        plt.ylabel("CV val MSE (standardised)")
        plt.title("Optuna search")
        plt.tight_layout()
        plt.savefig(out / "tuning_history.png", dpi=130)
        plt.close()


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.environ.get("FYP_DATA_DIR", "."))
    ap.add_argument("--multi", default="kse100_lstm_multi.csv")
    ap.add_argument("--features", default="kse100_lstm_features.csv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--params", default=None, help="json with hyper-parameters; skips tuning")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--timeout", type=int, default=None, help="tuning time limit in seconds")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--tune_window", action="store_true", help="also search window in {20,30,60}")
    ap.add_argument("--no_shrink", action="store_true", help="skip validation-fitted shrinkage toward drift")
    ap.add_argument("--seeds", type=int, default=10, help="ensemble members per model")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mc", type=int, default=100, help="MC dropout passes per ensemble member")
    ap.add_argument("--confidence", type=float, default=0.90)
    ap.add_argument("--tune_epochs", type=int, default=60)
    ap.add_argument("--tune_patience", type=int, default=8)
    ap.add_argument("--max_epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--no_ablation", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--quick", action="store_true", help="tiny smoke test")
    args = ap.parse_args()
    if args.quick:
        args.trials, args.folds, args.seeds, args.mc = 4, 2, 2, 20
        args.tune_epochs, args.max_epochs, args.patience, args.tune_patience = 8, 15, 5, 4

    warnings.filterwarnings("ignore", category=UserWarning)
    torch.backends.cudnn.deterministic = True
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    data_dir = Path(args.data_dir)
    out = Path(args.out) if args.out else data_dir / "lstm_outputs"
    out.mkdir(parents=True, exist_ok=True)
    log(f"device {device} | out {out}")

    data = Data(data_dir / args.multi)
    df = data.df
    for s in ("train", "val", "test", "purge"):
        r = df[df.split == s]
        log(f"{s:5s} {len(r):5d} rows  {r.Date.min().date()} -> {r.Date.max().date()}")

    # ---- 1. hyper-parameters ----
    study = None
    if args.params:
        params = json.load(open(args.params))
        params.setdefault("window", args.window)
        log("using params from file:", params)
    else:
        params, study = tune(data, args, device, out)
    window = params["window"]

    # ---- 2. final models ----
    seeds = [args.seed + i for i in range(args.seeds)]
    names = [MAIN] if args.no_ablation else [MAIN] + list(CONFIGS)
    log("selected features:", cols_from_params(params))
    ens = {}
    for n in names:
        ens[n] = fit_ensemble(n, params, data, seeds, device, args.max_epochs, args.patience, args.mc)

    # ---- 3. predictions (val for calibration, test for evaluation) ----
    va, te = data.rows(["val"], window), data.rows(["test"], window)
    close_te, Y_te, close_va, Y_va = data.close[te], data.Y[te], data.close[va], data.Y[va]
    tr = data.rows(["train"], window)
    drift = data.Y[tr].mean(0)
    mc_val, mc_test, mu_test, mu_raw, shrink = {}, {}, {}, {}, {}
    spread = []
    for n, e in ens.items():
        mc_val[n], mc_test[n] = predict(e, va), predict(e, te)
        mu_raw[n] = mc_test[n].mean(0)
        # per-member skill (spread across seeds), before shrinkage
        per = mc_test[n].reshape(len(e.models), -1, *mc_test[n].shape[1:]).mean(1)
        for j, h in enumerate(HORIZONS):
            sk = 1 - np.mean((per[:, :, j] - Y_te[:, j]) ** 2, axis=1) / np.mean(Y_te[:, j] ** 2)
            spread.append({"model": n, "horizon": h,
                           "ensemble_skill": 1 - np.mean((mu_raw[n][:, j] - Y_te[:, j]) ** 2) / np.mean(Y_te[:, j] ** 2),
                           "member_skill_mean": sk.mean(), "member_skill_std": sk.std(),
                           "member_skill_min": sk.min(), "member_skill_max": sk.max()})
        # shrink toward the train drift; factor fitted on validation only, clipped to [0, 1]
        mv0 = mc_val[n].mean(0)
        d, t = mv0 - drift, Y_va - drift
        s_ = np.clip((d * t).sum(0) / np.maximum((d * d).sum(0), 1e-12), 0.0, 1.0)
        shrink[n] = np.ones(H) if args.no_shrink else s_
        mc_val[n] = drift + shrink[n] * (mc_val[n] - drift)
        mc_test[n] = drift + shrink[n] * (mc_test[n] - drift)
        mu_test[n] = mc_test[n].mean(0)
    pd.DataFrame(spread).to_csv(out / "seed_spread.csv", index=False)
    log("shrinkage factors (1 = no shrink):", {n: np.round(v, 2).tolist() for n, v in shrink.items() if n in (MAIN, "full")})

    # ---- 4. baselines ----
    baselines = {"random_walk": np.zeros_like(Y_te),
                 "train_drift": np.tile(data.Y[tr].mean(0), (len(te), 1))}
    rp, alpha = ridge_baseline(data, window)
    baselines["ridge_full"] = rp
    log(f"ridge alpha {alpha:.3g}")

    rows = []
    for n, mu in baselines.items():
        rows += horizon_metrics(n, close_te, Y_te, mu)
    for n in ens:
        rows += horizon_metrics(f"lstm_{n}", close_te, Y_te, mu_test[n])
    rows += horizon_metrics(f"lstm_{MAIN}_noshrink", close_te, Y_te, mu_raw[MAIN])
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "test_metrics.csv", index=False)
    pd.set_option("display.width", 200)
    log("TEST metrics (price-space RMSE/MAE in index points; skill = 1 - MSE/MSE_randomwalk on log-returns)")
    print(metrics.round(4).to_string(index=False))

    # ---- 5. intervals for the full model ----
    f = ens[MAIN]
    mv, sv = mc_val[MAIN].mean(0), mc_val[MAIN].std(0)
    mt, st = mc_test[MAIN].mean(0), mc_test[MAIN].std(0)
    res_sd = np.sqrt(np.mean((mv - Y_va) ** 2, axis=0))          # validation residual scale per horizon
    tot_v, tot_t = np.sqrt(sv ** 2 + res_sd ** 2), np.sqrt(st ** 2 + res_sd ** 2)
    k = conformal_k(mv, tot_v, Y_va, args.confidence)
    z = norm_ppf((1 + args.confidence) / 2)
    cov_raw = np.mean(np.abs(Y_te - mt) <= z * st, axis=0)       # MC spread only
    cov_cal = np.mean(np.abs(Y_te - mt) <= k * tot_t, axis=0)
    lo_lr, hi_lr = mt - k * tot_t, mt + k * tot_t
    log(f"interval {args.confidence:.0%}: MC-only coverage {np.round(cov_raw, 3)} | "
        f"calibrated k {np.round(k, 2)} -> test coverage {np.round(cov_cal, 3)}")
    cov_df = pd.DataFrame({"horizon": HORIZONS, "nominal": args.confidence, "mc_only_coverage": cov_raw,
                           "calibrated_k": k, "calibrated_coverage": cov_cal,
                           "mean_width_pct": [100 * np.mean((np.exp(hi_lr[:, j]) - np.exp(lo_lr[:, j]))) for j in range(H)]})
    cov_df.to_csv(out / "interval_coverage.csv", index=False)

    pred = pd.DataFrame({"Date": df.Date.values[te], "Close": close_te})
    for j, h in enumerate(HORIZONS):
        pred[f"target_date_h{h}"] = df[f"target_date_h{h}"].values[te]
        pred[f"pred_logret_h{h}"] = mt[:, j]
        pred[f"pred_close_h{h}"] = close_te * np.exp(mt[:, j])
        pred[f"lo_h{h}"] = close_te * np.exp(lo_lr[:, j])
        pred[f"hi_h{h}"] = close_te * np.exp(hi_lr[:, j])
        pred[f"actual_close_h{h}"] = close_te * np.exp(Y_te[:, j])
    pred.to_csv(out / "test_predictions.csv", index=False)

    # ---- 6. ablation significance + per-regime errors ----
    if len(ens) > 1:
        dm_rows = []
        pairs = [(MAIN, "price_only"), (MAIN, "full"), (MAIN, "train_drift"), (MAIN, "random_walk"),
                 ("price_only", "random_walk"), ("full", "price_only"), ("full", "random_walk")]
        for a, b in pairs:
            for j, h in enumerate(HORIZONS):
                e1 = mu_test[a][:, j] - Y_te[:, j]
                e2 = (baselines[b][:, j] if b in baselines else mu_test[b][:, j]) - Y_te[:, j]
                stat, p = dm_test(e1, e2, h)
                dm_rows.append({"model_1": a, "model_2": b, "horizon": h, "DM_stat": stat, "p_value": p,
                                "better": a if stat < 0 else b})
        dm = pd.DataFrame(dm_rows)
        dm.to_csv(out / "dm_tests.csv", index=False)
        log("Diebold-Mariano (stat<0 => model_1 lower squared error)")
        print(dm.round(4).to_string(index=False))

        reg = np.select([df.Regime_Bull.values[te] == 1, df.Regime_Bear.values[te] == 1], ["bull", "bear"], "sideways")
        rr = []
        for n in (MAIN, "full", "price_only"):
            for rg in ("bull", "bear", "sideways"):
                m = reg == rg
                if m.sum() == 0:
                    continue
                for j, h in enumerate(HORIZONS):
                    rr.append({"model": n, "regime": rg, "n_days": int(m.sum()), "horizon": h,
                               "logret_RMSE": np.sqrt(np.mean((mu_test[n][m, j] - Y_te[m, j]) ** 2)),
                               "RW_logret_RMSE": np.sqrt(np.mean(Y_te[m, j] ** 2))})
        pd.DataFrame(rr).to_csv(out / "regime_errors.csv", index=False)

    # ---- 7. save final model + latest forecast ----
    torch.save({"states": [m.state_dict() for m in f.models], "params": params, "cols": f.cols,
                "scaler": f.scaler, "ym": f.ym, "ys": f.ys, "shrink": shrink[MAIN], "drift": drift, "calib_k": k, "resid_sd": res_sd, "confidence": args.confidence,
                "horizons": HORIZONS, "window": window}, out / "finsight_lstm.pt")
    json.dump(params, open(out / "best_params.json", "w"), indent=2)

    fpath = data_dir / args.features
    if fpath.exists():
        fd = pd.read_csv(fpath, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
        ff = make_features(fd)
        Xs = apply_scaler(ff, f.scaler)
        Wl = Xs[-window:][None]
        vs_last = (np.maximum(ff["roll_vol_20"].values[-1], 1e-3) * np.sqrt(np.array(HORIZONS, float))
                   if params.get("target_scale", "none") == "vol" else np.ones(H))
        S = (np.concatenate([mc_predict(m, Wl, args.mc, device) for m in f.models]) * f.ys + f.ym) * vs_last
        S = drift + shrink[MAIN] * (S - drift)
        mu = S.mean(0)[0]
        sd = np.sqrt(S.std(0)[0] ** 2 + res_sd ** 2)
        c0 = fd.Close.iloc[-1]
        lat = pd.DataFrame({"base_date": fd.Date.iloc[-1].date(), "base_close": c0, "horizon_trading_days": HORIZONS,
                            "forecast_close": c0 * np.exp(mu), "lower": c0 * np.exp(mu - k * sd),
                            "upper": c0 * np.exp(mu + k * sd), "expected_change_pct": 100 * (np.exp(mu) - 1)})
        lat.to_csv(out / "latest_forecast.csv", index=False)
        log("latest forecast"); print(lat.round(2).to_string(index=False))
    else:
        log(f"{fpath.name} not found, skipping latest forecast")

    make_plots(out, pred, f, metrics, study)
    log("done. Outputs in", out)


if __name__ == "__main__":
    main()
