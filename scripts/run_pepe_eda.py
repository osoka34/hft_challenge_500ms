#!/usr/bin/env python3
"""Run PEPE DQ / EDA / lead-lag / imbalance baseline; write reports + JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from hft_common import (  # noqa: E402
    ALL_DAYS,
    HORIZONS,
    POINTS_PER_DAY,
    TRAIN_DAYS,
    VAL_DAYS,
    bbo_mid_series,
    day_start_ms,
    load_day_pair,
    log_returns,
    read_parquet,
    reindex_mid_grid,
)
from hft_metric import (  # noqa: E402
    apply_imbalance_baseline,
    daily_scores,
    fit_clip_bounds,
    fit_imbalance_baseline,
    mean_score,
    ols_fit,
)

OUT_JSON = ROOT / "outputs" / "pepe_eda_results.json"
REPORT_LOG = ROOT / "reports" / "VARIABLE_ANALYSIS_LOG.md"
PLAN = ROOT / "reports" / "HFT_ANALYSIS_PLAN.md"


def dq_one_day(day: str) -> dict:
    start = day_start_ms(day)
    out = {"day": day}
    for inst, streams in {
        "1000PEPEUSDT_PERP": ("book", "bbo", "trades", "targets"),
        "1000PEPEUSDC_PERP": ("book", "bbo", "trades"),
    }.items():
        for stream in streams:
            df = read_parquet(inst, stream, day)
            ts = df["ts"].to_numpy(dtype=np.int64)
            key = f"{inst}:{stream}"
            info = {
                "n": int(len(df)),
                "first_lag_ms": int(ts.min() - start) if len(ts) else None,
                "last_lag_ms": int((start + 86_400_000 - 1) - ts.max()) if len(ts) else None,
            }
            if len(ts) >= 2:
                gaps = np.diff(np.sort(ts))
                info["max_gap_ms"] = int(gaps.max())
                info["median_gap_ms"] = float(np.median(gaps))
            if stream == "book":
                locked = (df["bid_px"] == df["ask_px"]).mean()
                crossed = (df["bid_px"] > df["ask_px"]).mean()
                info["locked_frac"] = float(locked)
                info["crossed_frac"] = float(crossed)
                info["missing_vs_full"] = int(POINTS_PER_DAY - len(df))
            if stream == "targets":
                for h in HORIZONS:
                    y = df[h].to_numpy(dtype=np.float64)
                    info[f"{h}_nan"] = int(np.isnan(y).sum())
                    finite = y[np.isfinite(y)]
                    info[f"{h}_zero_frac"] = float((finite == 0).mean()) if len(finite) else None
                    info[f"{h}_std_bp"] = float(np.nanstd(y) * 1e4)
            out[key] = info
    return out


def imbalance_ic(frames: dict[str, pd.DataFrame], days: list[str]) -> dict:
    rows = []
    for day in days:
        df = frames[day]
        imb = df["imb"].to_numpy(dtype=np.float64)
        for h in HORIZONS:
            y = df[h].to_numpy(dtype=np.float64)
            mask = np.isfinite(imb) & np.isfinite(y)
            if mask.sum() < 100:
                continue
            # Spearman via rank pearson
            xr = pd.Series(imb[mask]).rank().to_numpy()
            yr = pd.Series(y[mask]).rank().to_numpy()
            xr = xr - xr.mean()
            yr = yr - yr.mean()
            den = np.sqrt((xr * xr).sum() * (yr * yr).sum())
            ic = float((xr * yr).sum() / den) if den else float("nan")
            rows.append({"day": day, "horizon": h, "spearman": ic, "ext_frac": float((np.abs(imb[mask]) > 0.9).mean())})
    tab = pd.DataFrame(rows)
    summary = {
        "mean_spearman": {h: float(tab.loc[tab.horizon == h, "spearman"].mean()) for h in HORIZONS},
        "mean_ext_frac": float(tab["ext_frac"].mean()) if len(tab) else None,
    }
    # S-shape: mean y by imb bin on pooled days
    pool = pd.concat([frames[d] for d in days], ignore_index=True)
    pool = pool[np.isfinite(pool["imb"]) & np.isfinite(pool["y_500ms"])]
    bins = np.linspace(-1, 1, 21)
    pool["bin"] = pd.cut(pool["imb"], bins=bins, include_lowest=True)
    sshape = pool.groupby("bin", observed=True)["y_500ms"].mean().mul(1e4)  # bp
    summary["sshape_y500_bp"] = {str(k): float(v) for k, v in sshape.items()}
    return summary


def cross_corr_leadlag(day: str, step_ms: int = 100, max_lag_steps: int = 20) -> dict:
    usdt = log_returns(reindex_mid_grid(bbo_mid_series("1000PEPEUSDT_PERP", day), day, step_ms))
    usdc = log_returns(reindex_mid_grid(bbo_mid_series("1000PEPEUSDC_PERP", day), day, step_ms))
    a = usdc.to_numpy(dtype=np.float64)
    b = usdt.to_numpy(dtype=np.float64)
    lags = list(range(0, max_lag_steps + 1))
    corrs = []
    for k in lags:
        if k == 0:
            x, y = a, b
        else:
            x, y = a[:-k], b[k:]
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 100:
            corrs.append(float("nan"))
            continue
        xx, yy = x[mask], y[mask]
        xx = xx - xx.mean()
        yy = yy - yy.mean()
        den = np.sqrt((xx * xx).sum() * (yy * yy).sum())
        corrs.append(float((xx * yy).sum() / den) if den else float("nan"))
    # reverse: USDT leads USDC
    rev = []
    for k in lags:
        if k == 0:
            x, y = b, a
        else:
            x, y = b[:-k], a[k:]
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 100:
            rev.append(float("nan"))
            continue
        xx, yy = x[mask], y[mask]
        xx = xx - xx.mean()
        yy = yy - yy.mean()
        den = np.sqrt((xx * xx).sum() * (yy * yy).sum())
        rev.append(float((xx * yy).sum() / den) if den else float("nan"))
    return {
        "step_ms": step_ms,
        "lag_ms": [k * step_ms for k in lags],
        "usdc_leads_usdt": corrs,
        "usdt_leads_usdc": rev,
        "peak_forward": {
            "lag_ms": int(lags[int(np.nanargmax(corrs))] * step_ms) if np.any(np.isfinite(corrs)) else None,
            "corr": float(np.nanmax(corrs)) if np.any(np.isfinite(corrs)) else None,
        },
    }


def gap_ic(frames: dict[str, pd.DataFrame], days: list[str], window: int = 120) -> dict:
    """gap_dev = gap - causal rolling mean over `window` steps of 500 ms (default 60 s)."""
    rows = []
    for day in days:
        df = frames[day].copy()
        gap = df["gap"].to_numpy(dtype=np.float64)
        # causal rolling mean
        mean = pd.Series(gap).rolling(window=window, min_periods=max(10, window // 5)).mean().shift(1)
        gap_dev = gap - mean.to_numpy()
        for h in HORIZONS:
            y = df[h].to_numpy(dtype=np.float64)
            mask = np.isfinite(gap_dev) & np.isfinite(y)
            if mask.sum() < 100:
                continue
            xr = pd.Series(gap_dev[mask]).rank().to_numpy()
            yr = pd.Series(y[mask]).rank().to_numpy()
            xr = xr - xr.mean()
            yr = yr - yr.mean()
            den = np.sqrt((xr * xr).sum() * (yr * yr).sum())
            ic = float((xr * yr).sum() / den) if den else float("nan")
            # also raw USDC return over last 500ms on book grid
            rows.append({"day": day, "horizon": h, "gap_dev_spearman": ic})
        # USDC mid return last step vs y
        mid_u = df["mid_usdc"].to_numpy(dtype=np.float64)
        r_usdc = np.full_like(mid_u, np.nan)
        r_usdc[1:] = np.log(mid_u[1:] / mid_u[:-1])
        y = df["y_500ms"].to_numpy(dtype=np.float64)
        mask = np.isfinite(r_usdc) & np.isfinite(y)
        if mask.sum() > 100:
            xr = pd.Series(r_usdc[mask]).rank().to_numpy()
            yr = pd.Series(y[mask]).rank().to_numpy()
            xr = xr - xr.mean()
            yr = yr - yr.mean()
            den = np.sqrt((xr * xr).sum() * (yr * yr).sum())
            rows.append(
                {
                    "day": day,
                    "horizon": "y_500ms",
                    "usdc_ret_spearman": float((xr * yr).sum() / den) if den else float("nan"),
                }
            )
    tab = pd.DataFrame(rows)
    out = {}
    for h in HORIZONS:
        s = tab.loc[tab.horizon == h, "gap_dev_spearman"].dropna()
        out[f"gap_dev_{h}"] = float(s.mean()) if len(s) else None
    s = tab["usdc_ret_spearman"].dropna() if "usdc_ret_spearman" in tab else pd.Series(dtype=float)
    out["usdc_ret_y500"] = float(s.mean()) if len(s) else None
    return out


def run_baseline(frames: dict[str, pd.DataFrame]) -> dict:
    train = [frames[d] for d in TRAIN_DAYS]
    bounds = fit_clip_bounds(train)
    models = fit_imbalance_baseline(train, bounds)
    pred_frames = {}
    for d, df in frames.items():
        pred_frames[d] = apply_imbalance_baseline(df, models)
    pred_cols = {h: f"pred_{h}" for h in HORIZONS}
    train_daily = daily_scores(TRAIN_DAYS, pred_frames, pred_cols, bounds)
    val_daily = daily_scores(VAL_DAYS, pred_frames, pred_cols, bounds)
    # M1: imb + imb^3 + own ret 0.5s / 2s
    m1_models = {}
    for h in HORIZONS:
        xs, ys = [], []
        for df in train:
            imb = df["imb"].to_numpy(dtype=np.float64)
            mid = df["mid"].to_numpy(dtype=np.float64)
            r1 = np.full_like(mid, np.nan)
            r1[1:] = np.log(mid[1:] / mid[:-1])
            r4 = np.full_like(mid, np.nan)
            r4[4:] = np.log(mid[4:] / mid[:-4])
            X = np.column_stack([np.ones(len(imb)), imb, imb**3, r1, r4])
            y = df[h].to_numpy(dtype=np.float64)
            y = np.clip(y, bounds.lo[h], bounds.hi[h])
            mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
            xs.append(X[mask])
            ys.append(y[mask])
        X = np.vstack(xs)
        y = np.concatenate(ys)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        m1_models[h] = coef
    for d, df in pred_frames.items():
        mid = df["mid"].to_numpy(dtype=np.float64)
        imb = df["imb"].to_numpy(dtype=np.float64)
        r1 = np.full_like(mid, np.nan)
        r1[1:] = np.log(mid[1:] / mid[:-1])
        r4 = np.full_like(mid, np.nan)
        r4[4:] = np.log(mid[4:] / mid[:-4])
        X = np.column_stack([np.ones(len(imb)), imb, imb**3, r1, r4])
        for h, coef in m1_models.items():
            pred_frames[d][f"m1_{h}"] = X @ coef
    m1_cols = {h: f"m1_{h}" for h in HORIZONS}
    m1_val = daily_scores(VAL_DAYS, pred_frames, m1_cols, bounds)
    return {
        "clip_lo": bounds.lo,
        "clip_hi": bounds.hi,
        "ols_coef": {h: {"a": models[h][0], "b": models[h][1]} for h in HORIZONS},
        "train": mean_score(train_daily),
        "val": mean_score(val_daily),
        "val_daily": val_daily.to_dict(orient="records"),
        "m1_val": mean_score(m1_val),
        "m1_val_daily": m1_val.to_dict(orient="records"),
    }


def write_log(results: dict) -> None:
    dq = results["dq"]
    # aggregate DQ
    book_rows = [d["1000PEPEUSDT_PERP:book"]["n"] for d in dq]
    first_lags = [d["1000PEPEUSDT_PERP:book"]["first_lag_ms"] for d in dq]
    max_gaps = [d["1000PEPEUSDT_PERP:book"]["max_gap_ms"] for d in dq]
    zero500 = [d["1000PEPEUSDT_PERP:targets"]["y_500ms_zero_frac"] for d in dq]
    zero2 = [d["1000PEPEUSDT_PERP:targets"]["y_2s_zero_frac"] for d in dq]
    zero8 = [d["1000PEPEUSDT_PERP:targets"]["y_8s_zero_frac"] for d in dq]
    std2 = [d["1000PEPEUSDT_PERP:targets"]["y_2s_std_bp"] for d in dq]
    locked = [d["1000PEPEUSDT_PERP:book"]["locked_frac"] for d in dq]
    crossed = [d["1000PEPEUSDT_PERP:book"]["crossed_frac"] for d in dq]
    nan_tot = sum(d["1000PEPEUSDT_PERP:targets"][f"{h}_nan"] for d in dq for h in HORIZONS)

    ll_train = results["leadlag"]["train_median_100ms"]
    ll_val = results["leadlag"]["val_median_100ms"]
    ll5_train = results["leadlag"]["train_median_500ms"]
    ll5_val = results["leadlag"]["val_median_500ms"]
    base = results["baseline"]

    text = f"""# Лог разбора переменных (HFT / PEPE start)

**Зачем этот файл.** Цифры и короткий вердикт.  
**Пояснения** — `EXPLANATIONS.md`.  
**План** — `HFT_ANALYSIS_PLAN.md`.  
**Код** — `notebooks/01_eda_pepe.ipynb`, `scripts/run_pepe_eda.py`.

**Данные.** Sample days train `{TRAIN_DAYS[0]}…{TRAIN_DAYS[-1]}` ({len(TRAIN_DAYS)}d), val `{VAL_DAYS[0]}…{VAL_DAYS[-1]}` ({len(VAL_DAYS)}d).  
JSON dump: `outputs/pepe_eda_results.json`.

---

## Roadmap

| # | Блок | Статус |
|---|------|--------|
| V01 | Data quality | **done** |
| V02 | Target zeros / daily vol | **done** |
| V03 | Own imbalance → y | **done** |
| V04 | PEPEUSDC → PEPEUSDT lead–lag | **done** |
| V05 | Same-coin mid gap | **done** |
| V06 | OLS imbalance baseline (PEPE-only) | **done** |

---

## V01 — Data quality

Статус: **done**

Цифры (USDT book/targets, {len(dq)} sample days):
- book rows / day: min {min(book_rows)}, median {int(np.median(book_rows))}, max {max(book_rows)} (full grid 172800)
- first book lag from 00:00 UTC: min {min(first_lags)} ms, median {int(np.median(first_lags))} ms, max {max(first_lags)} ms
- max intra-day book gap: min {min(max_gaps)} ms, median {int(np.median(max_gaps))} ms, max {max(max_gaps)} ms
- locked quotes frac (median): {float(np.median(locked)):.4f}; crossed (median): {float(np.median(crossed)):.6f}
- target NaN count across sample×3 horizons: {nan_tot} (ожидаемо у открытых точек дня)

Вывод: файлы чистые; день стартует с опозданием; дыры редкие на sample. **Keep** late-start / gap hygiene.

---

## V02 — Target zeros / daily vol

Статус: **done**

Цифры (доля exact zero, median по дням):
- y_500ms: {float(np.median(zero500)):.3f}
- y_2s: {float(np.median(zero2)):.3f}
- y_8s: {float(np.median(zero8)):.3f}
- daily std y_2s: median {float(np.median(std2)):.2f} bp (min {min(std2):.2f}, max {max(std2):.2f})

Вывод: нули доминируют на коротком горизонте — как у peers. **Keep** как контекст потолка corr.

---

## V03 — Own imbalance → y

Статус: **done**

Цифры (mean Spearman IC по дням):
- train: {json.dumps(results['imb_ic']['train']['mean_spearman'])}
- val: {json.dumps(results['imb_ic']['val']['mean_spearman'])}
- доля |imb|>0.9 (mean): train {results['imb_ic']['train']['mean_ext_frac']:.3f}, val {results['imb_ic']['val']['mean_ext_frac']:.3f}

Вывод: imbalance связан с forward y; экстремумы часты на PEPE. **Keep** как ядро M0.

---

## V04 — PEPEUSDC → PEPEUSDT lead–lag

Статус: **done**

Цифры — 100 ms grid, median corr(r_USDC(t−k), r_USDT(t)); peak только по k>0 (lag 0 = contemporaneous):
- contemporaneous (k=0) train/val: {ll_train['corr_0']:.4f} / {ll_val['corr_0']:.4f}
- train peak k>0: lag {ll_train['peak_lag_ms']} ms, corr {ll_train['peak_corr']:.4f}
- val peak k>0: lag {ll_val['peak_lag_ms']} ms, corr {ll_val['peak_corr']:.4f}
- train corr at 100/300 ms: {ll_train['corr_100']:.4f} / {ll_train['corr_300']:.4f} · reverse@100: {ll_train['rev_corr_100']:.4f}
- val corr at 100/300 ms: {ll_val['corr_100']:.4f} / {ll_val['corr_300']:.4f} · reverse@100: {ll_val['rev_corr_100']:.4f}

Цифры — 500 ms grid (book mids via bbo reindex), same definition:
- contemporaneous train/val: {ll5_train['corr_0']:.4f} / {ll5_val['corr_0']:.4f}
- train peak k>0: lag {ll5_train['peak_lag_ms']} ms, corr {ll5_train['peak_corr']:.4f}
- val peak k>0: lag {ll5_val['peak_lag_ms']} ms, corr {ll5_val['peak_corr']:.4f}
- train@500/1000 ms: {ll5_train['corr_500']:.4f} / {ll5_train['corr_1000']:.4f} · reverse@500: {ll5_train['rev_corr_500']:.4f}
- val@500/1000 ms: {ll5_val['corr_500']:.4f} / {ll5_val['corr_1000']:.4f} · reverse@500: {ll5_val['rev_corr_500']:.4f}

Вывод: рынки почти синхронны (высокий corr@0). На 100 ms сетке USDC→USDT @100 ms сильнее reverse на train; на val lead слабее. **Keep** lagged USDC returns (≤300–500 ms).

---

## V05 — Same-coin mid gap

Статус: **done**

Цифры (mean Spearman):
- gap_dev (60s mean) → horizons train: {json.dumps({k: results['gap_ic']['train'].get(k) for k in results['gap_ic']['train'] if k.startswith('gap_dev')})}
- gap_dev val: {json.dumps({k: results['gap_ic']['val'].get(k) for k in results['gap_ic']['val'] if k.startswith('gap_dev')})}
- USDC last-500ms return → y_500ms: train {results['gap_ic']['train'].get('usdc_ret_y500')}, val {results['gap_ic']['val'].get('usdc_ret_y500')}

Вывод: `|gap_dev|` IC сопоставим с USDC return (знак отрицательный = mean-reversion). **Keep** оба как M2 candidates.

---

## V06 — OLS imbalance baseline (PEPE-only SCORE)

Статус: **done**

Метрика: clip по train 0.1%/99.9% → daily Pearson → mean по 3 горизонтам PEPE (не полный 12-cell SCORE).

Цифры:
- M0 imb OLS train mean: {base['train']['mean']:.4f} · by H: {json.dumps(base['train']['by_horizon'])}
- M0 imb OLS val mean: **{base['val']['mean']:.4f}** · by H: {json.dumps(base['val']['by_horizon'])}
- M1 (imb, imb³, own r 0.5s/2s) val mean: **{base['m1_val']['mean']:.4f}** · by H: {json.dumps(base['m1_val']['by_horizon'])}
- Peer PEPE baseline cells ~0.15/0.16/0.12 (full val 62d) — наш sample короче, порядок должен быть похож.

Вывод: пайплайн метрики работает; M1 даёт небольшой lift. Дальше — depth/OFI/USDC features после догрузки остальных perps для полного SCORE.

---

## Hypotheses

| id | Claim | EDA support | Strength | Next |
|----|-------|-------------|----------|------|
| H1 | Own L1 imbalance forecasts PEPE short-horizon y | IC + baseline SCORE | strong | Keep in all models |
| H2 | Lagged PEPE·USDC returns help PEPE·USDT (≤300–500 ms) | corr@100ms > reverse on train; val weaker | medium–strong | Add lagged USDC returns |
| H3 | gap_dev mean-reverts (negative IC) | |IC|~0.11–0.15 | medium | Keep as M2 candidate |
| H4 | Zeros dominate y_500ms → corr ceiling lower than 8s | zero fractions | strong | Don't over-weight 500ms alone |
"""
    REPORT_LOG.write_text(text)
    print(f"wrote {REPORT_LOG}")


def summarize_leadlag(curves: list[dict]) -> dict:
    mats = np.array([c["usdc_leads_usdt"] for c in curves], dtype=float)
    revs = np.array([c["usdt_leads_usdc"] for c in curves], dtype=float)
    med = np.nanmedian(mats, axis=0)
    med_rev = np.nanmedian(revs, axis=0)
    lags = curves[0]["lag_ms"]

    def at(ms: int) -> float:
        if ms not in lags:
            return float("nan")
        return float(med[lags.index(ms)])

    def at_rev(ms: int) -> float:
        if ms not in lags:
            return float("nan")
        return float(med_rev[lags.index(ms)])

    # lag 0 = contemporaneous; predictive lead is k > 0
    pos = np.array(lags) > 0
    peak_i = int(np.nanargmax(np.where(pos, med, -np.inf)))
    rev_i = int(np.nanargmax(np.where(pos, med_rev, -np.inf)))
    return {
        "corr_0": at(0),
        "peak_lag_ms": int(lags[peak_i]),
        "peak_corr": float(med[peak_i]),
        "corr_100": at(100),
        "corr_300": at(300),
        "corr_500": at(500),
        "corr_1000": at(1000),
        "rev_peak_lag_ms": int(lags[rev_i]),
        "rev_peak_corr": float(med_rev[rev_i]),
        "rev_corr_100": at_rev(100),
        "rev_corr_500": at_rev(500),
        "median_curve": med.tolist(),
        "median_reverse": med_rev.tolist(),
        "lag_ms": lags,
    }


def main() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    print("loading pair frames…")
    frames = {d: load_day_pair(d) for d in ALL_DAYS}
    print("DQ…")
    dq = [dq_one_day(d) for d in ALL_DAYS]
    print("imbalance IC…")
    imb_ic = {
        "train": imbalance_ic(frames, TRAIN_DAYS),
        "val": imbalance_ic(frames, VAL_DAYS),
    }
    print("lead-lag 100ms…")
    ll_train_100 = [cross_corr_leadlag(d, step_ms=100, max_lag_steps=20) for d in TRAIN_DAYS]
    ll_val_100 = [cross_corr_leadlag(d, step_ms=100, max_lag_steps=20) for d in VAL_DAYS]
    print("lead-lag 500ms…")
    ll_train_500 = [cross_corr_leadlag(d, step_ms=500, max_lag_steps=10) for d in TRAIN_DAYS]
    ll_val_500 = [cross_corr_leadlag(d, step_ms=500, max_lag_steps=10) for d in VAL_DAYS]
    leadlag = {
        "train_median_100ms": summarize_leadlag(ll_train_100),
        "val_median_100ms": summarize_leadlag(ll_val_100),
        "train_median_500ms": summarize_leadlag(ll_train_500),
        "val_median_500ms": summarize_leadlag(ll_val_500),
    }
    print("gap IC…")
    gap = {"train": gap_ic(frames, TRAIN_DAYS), "val": gap_ic(frames, VAL_DAYS)}
    print("baseline…")
    baseline = run_baseline(frames)
    results = {
        "days": {"train": TRAIN_DAYS, "val": VAL_DAYS},
        "dq": dq,
        "imb_ic": imb_ic,
        "leadlag": leadlag,
        "gap_ic": gap,
        "baseline": baseline,
    }
    # slim JSON (drop per-day leadlag curves — already medianized)
    OUT_JSON.write_text(json.dumps(results, indent=2, default=float))
    print(f"wrote {OUT_JSON}")
    write_log(results)
    print("val M0", baseline["val"]["mean"], "M1", baseline["m1_val"]["mean"])


if __name__ == "__main__":
    main()
