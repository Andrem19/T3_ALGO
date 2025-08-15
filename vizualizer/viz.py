# profit_cor_viz.py
# -*- coding: utf-8 -*-
"""
Стратифицированная визуализация и анализ корреляций profit с параметрами опционов.

Ключевые отличия:
  • Строгое разделение: сначала по opt_type (P/C), затем внутри по opt_mon (ATM/OTM/ITM).
  • 2D-теплокарты среднего profit для пар:
        (opt_index, q_frac), (opt_index, gamma), (q_frac, duration)
    — отдельно по type_of_signal = 1 и 2, и общий по обоим.
  • Частные корреляции: corr(profit, feature | type_of_signal, opt_type) — через резидуализацию.
  • Сравнение распределений profit для сигналов 1 и 2 (ECDF и QQ-plot).
  • Лёгкое моделирование CatBoost/XGBoost (если доступны) с fallback на sklearn;
    сохраняем важности признаков (и SHAP, если установлен shap).

Ограничения:
  • matplotlib only; без seaborn; один график — одна фигура.
  • Все результаты пишутся в out_dir; формируется понятная структура подпапок.

Использование:
    from profit_cor_viz import run_full_analysis
    run_full_analysis(trades, out_dir="_viz_statistic")
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _bin_quantiles(x: pd.Series, q: int = 10) -> pd.Categorical:
    x = pd.to_numeric(x, errors="coerce")
    uniq = x.dropna().unique()
    if uniq.size < 3:
        bins = max(min(uniq.size, 3), 1)
    else:
        bins = min(q, uniq.size)
    try:
        return pd.qcut(x, q=bins, duplicates="drop")
    except Exception:
        return pd.cut(x, bins=1)


def _rank(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    ranks = np.full(a.shape, np.nan)
    mask = ~np.isnan(a)
    if mask.sum() < 1:
        return ranks
    vals = a[mask]
    order = np.argsort(vals, kind="mergesort")
    sorted_vals = vals[order]
    out = np.empty_like(sorted_vals, dtype=float)
    i = 0
    while i < len(sorted_vals):
        j = i + 1
        while j < len(sorted_vals) and sorted_vals[j] == sorted_vals[i]:
            j += 1
        out[i:j] = (i + 1 + j) / 2.0
        i = j
    ranks[mask] = out
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(float); y = y.astype(float)
    m = ~np.isnan(x) & ~np.isnan(y)
    if m.sum() < 2: return float("nan")
    if np.std(x[m], ddof=1) == 0 or np.std(y[m], ddof=1) == 0:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    xr = _rank(np.asarray(x, dtype=float))
    yr = _rank(np.asarray(y, dtype=float))
    return _pearson(xr, yr)


def _partial_corr_pearson(y: np.ndarray, x: np.ndarray, controls: np.ndarray) -> float:
    def ols_resid(t: np.ndarray, C: np.ndarray) -> np.ndarray:
        m = ~np.isnan(t) & ~np.isnan(C).any(axis=1)
        if m.sum() < C.shape[1] + 1:
            return np.full_like(t, np.nan, dtype=float)
        Cm = C[m]
        tm = t[m]
        beta = np.linalg.pinv(Cm) @ tm
        resid = np.full_like(t, np.nan, dtype=float)
        resid[m] = tm - Cm @ beta
        return resid

    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    C = np.asarray(controls, dtype=float)
    if C.ndim == 1:
        C = C.reshape(-1, 1)
    C = np.column_stack([np.ones(C.shape[0]), C])
    ry = ols_resid(y, C)
    rx = ols_resid(x, C)
    return _pearson(ry, rx)


def flatten_trades(trades: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for i, tr in enumerate(trades):
        profit = _to_float(tr.get("profit"))
        duration = _to_float(tr.get("duration"))
        type_of_close = tr.get("type_of_close")
        type_of_signal = _to_float(tr.get("type_of_signal"))
        for sp in tr.get("snap_params") or []:
            row = {
                "trade_idx": i,
                "profit": profit,
                "duration": duration,
                "type_of_close": type_of_close,
                "type_of_signal": type_of_signal,
                "opt_type": sp.get("opt_type"),
                "opt_mon": sp.get("opt_mon"),
                "over": _to_float(sp.get("over")),
                "opt_index": _to_float(sp.get("opt_index")),
                "q_frac": _to_float(sp.get("q_frac")),
                "gamma": _to_float(sp.get("gamma")),
                "delta": _to_float(sp.get("delta")),
            }
            row["abs_delta"] = abs(row["delta"]) if row["delta"] is not None else None
            rows.append(row)
    df = pd.DataFrame(rows)
    for c in ["profit", "duration", "type_of_signal", "over", "opt_index", "q_frac", "gamma", "delta", "abs_delta"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _heatmap_2d_mean_profit(df: pd.DataFrame, x_col: str, y_col: str, out_path: Path, title: str, bins: int = 10) -> None:
    x_bins = _bin_quantiles(df[x_col], q=bins)
    y_bins = _bin_quantiles(df[y_col], q=bins)
    tmp = pd.DataFrame({"xb": x_bins, "yb": y_bins, "profit": df["profit"]}).dropna()
    if tmp.empty:
        return
    pivot = tmp.pivot_table(index="yb", columns="xb", values="profit", aggfunc="mean")

    def centers(cats: pd.Categorical) -> list:
        res = []
        for iv in cats.cat.categories:
            try:
                res.append((iv.left + iv.right) / 2.0)
            except Exception:
                res.append(np.nan)
        return res

    x_centers = centers(x_bins)
    y_centers = centers(y_bins)

    plt.figure(figsize=(8, 7))
    z = pivot.values
    plt.imshow(z, aspect="auto", origin="lower", interpolation="nearest")
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.colorbar(label="Mean profit")
    xticks = np.linspace(0, z.shape[1] - 1, min(10, z.shape[1]), dtype=int)
    yticks = np.linspace(0, z.shape[0] - 1, min(10, z.shape[0]), dtype=int)
    plt.xticks(xticks, [f"{x_centers[i]:.4g}" if i < len(x_centers) else "" for i in xticks], rotation=45, ha="right")
    plt.yticks(yticks, [f"{y_centers[i]:.4g}" if i < len(y_centers) else "" for i in yticks])
    _save_fig(out_path)


def build_stratified_heatmaps(df: pd.DataFrame, out_dir: Path) -> None:
    _ensure_dir(out_dir)
    pairs = [("opt_index", "q_frac"), ("opt_index", "gamma"), ("q_frac", "duration")]
    opt_types = [v for v in ["P", "C"] if v in df["opt_type"].dropna().unique().tolist()]
    opt_mons = df["opt_mon"].dropna().unique().tolist()

    for ot in opt_types:
        for om in opt_mons:
            df_stratum = df[(df["opt_type"] == ot) & (df["opt_mon"] == om)].copy()
            if df_stratum.empty:
                continue
            base_dir = out_dir / f"{ot}" / f"{om}"
            _ensure_dir(base_dir)

            for xcol, ycol in pairs:
                title = f"Mean profit heatmap ALL | {ot} / {om} | {xcol} vs {ycol}"
                _heatmap_2d_mean_profit(df_stratum, xcol, ycol, base_dir / f"heatmap_{xcol}_{ycol}_ALL.png", title)

            for sig in [1.0, 2.0]:
                dfs = df_stratum[df_stratum["type_of_signal"] == sig]
                if dfs.empty:
                    continue
                for xcol, ycol in pairs:
                    title = f"Mean profit heatmap signal={int(sig)} | {ot} / {om} | {xcol} vs {ycol}"
                    _heatmap_2d_mean_profit(dfs, xcol, ycol, base_dir / f"heatmap_{xcol}_{ycol}_sig{int(sig)}.png", title)


def compute_partial_correlations(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    _ensure_dir(out_dir)
    y = pd.to_numeric(df["profit"], errors="coerce").values
    sig = pd.to_numeric(df["type_of_signal"], errors="coerce").values.reshape(-1, 1)
    opt_type_num = (df["opt_type"] == "C").astype(float).values.reshape(-1, 1)
    controls = np.hstack([sig, opt_type_num])

    features = ["over", "opt_index", "q_frac", "gamma", "delta", "abs_delta", "duration"]
    recs = []
    for f in features:
        x = pd.to_numeric(df[f], errors="coerce").values
        r = _partial_corr_pearson(y, x, controls)
        recs.append({"feature": f, "partial_pearson_r": r})
    out = pd.DataFrame(recs).set_index("feature")
    out.to_csv(out_dir / "partial_correlations.csv", float_format="%.6f")
    return out


def plot_ecdf_profit_by_signal(df: pd.DataFrame, out_path: Path) -> None:
    d1 = pd.to_numeric(df.loc[df["type_of_signal"] == 1, "profit"], errors="coerce").dropna().values
    d2 = pd.to_numeric(df.loc[df["type_of_signal"] == 2, "profit"], errors="coerce").dropna().values
    if d1.size == 0 and d2.size == 0:
        return
    plt.figure(figsize=(8, 6))
    for arr, lab in [(d1, "signal=1"), (d2, "signal=2")]:
        if arr.size == 0: 
            continue
        x = np.sort(arr); y = np.arange(1, len(x) + 1) / len(x)
        plt.plot(x, y, label=lab)
    plt.title("ECDF: распределение profit по type_of_signal")
    plt.xlabel("profit")
    plt.ylabel("F(x)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save_fig(out_path)


def plot_qq_profit_by_signal(df: pd.DataFrame, out_path: Path, n_quant: int = 200) -> None:
    d1 = pd.to_numeric(df.loc[df["type_of_signal"] == 1, "profit"], errors="coerce").dropna().values
    d2 = pd.to_numeric(df.loc[df["type_of_signal"] == 2, "profit"], errors="coerce").dropna().values
    if d1.size == 0 or d2.size == 0:
        return
    qs = np.linspace(0.01, 0.99, n_quant)
    q1 = np.quantile(d1, qs); q2 = np.quantile(d2, qs)
    plt.figure(figsize=(7, 7))
    plt.scatter(q1, q2, s=14, alpha=0.7)
    mn = min(q1.min(), q2.min()); mx = max(q1.max(), q2.max())
    xs = np.linspace(mn, mx, 200)
    plt.plot(xs, xs, linewidth=2.0)
    plt.title("QQ-plot: profit (signal=1) vs (signal=2)")
    plt.xlabel("Квантили profit при signal=1")
    plt.ylabel("Квантили profit при signal=2")
    plt.grid(True, alpha=0.3)
    _save_fig(out_path)


def _prepare_ml_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    y = pd.to_numeric(df["profit"], errors="coerce")
    base_cols = ["over", "opt_index", "q_frac", "gamma", "duration", "abs_delta"]
    X = df[base_cols].copy()
    X["type_of_signal"] = pd.to_numeric(df["type_of_signal"], errors="coerce")
    X["opt_type_is_C"] = (df["opt_type"] == "C").astype(float)
    opt_mon_dummies = pd.get_dummies(df["opt_mon"].astype(str), prefix="opt_mon")
    X = pd.concat([X, opt_mon_dummies], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    y = pd.to_numeric(y, errors="coerce")
    mask = y.notna()
    X = X.loc[mask].copy(); y = y.loc[mask].copy()
    features = X.columns.tolist()
    return X, y, features


def train_model_and_feature_importance(df: pd.DataFrame, out_dir: Path) -> None:
    _ensure_dir(out_dir)
    X, y, features = _prepare_ml_frame(df)

    model = None; model_name = None; trained = False; warn_msg = None
    try:
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(depth=6, learning_rate=0.05, n_estimators=500, loss_function="RMSE",
                                  random_seed=42, verbose=False)
        model.fit(X, y); model_name = "CatBoostRegressor"; trained = True
    except Exception as e:
        warn_msg = f"CatBoost недоступен: {e}"
    if not trained:
        try:
            import xgboost as xgb
            model = xgb.XGBRegressor(n_estimators=800, max_depth=6, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=2)
            model.fit(X, y); model_name = "XGBRegressor"; trained = True
        except Exception as e:
            warn_msg = (warn_msg or "") + f" | XGBoost недоступен: {e}"
    if not trained:
        try:
            from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
        except Exception:
            pass
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
            model = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=800, random_state=42)
            model.fit(X, y); model_name = "HistGradientBoostingRegressor"; trained = True
        except Exception as e:
            warn_msg = (warn_msg or "") + f" | Sklearn HGB недоступен: {e}"
    if not trained:
        try:
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=1.0, random_state=42)
            model.fit(X, y); model_name = "Ridge"; trained = True
        except Exception as e:
            warn_msg = (warn_msg or "") + f" | Ridge недоступен: {e}"

    if not trained:
        with open(out_dir / "model_error.txt", "w", encoding="utf-8") as f:
            f.write("Не удалось обучить модель. " + (warn_msg or ""))
        return

    importances = None
    try:
        if hasattr(model, "feature_importances_"):
            importances = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "coef_"):
            importances = np.abs(np.asarray(model.coef_, dtype=float))
    except Exception:
        importances = None

    if importances is not None and importances.size == len(features):
        order = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 7))
        plt.bar(range(len(features)), importances[order])
        plt.xticks(range(len(features)), [features[i] for i in order], rotation=45, ha="right")
        plt.title(f"Важности признаков ({model_name})")
        plt.ylabel("Importance (model-based)")
        plt.grid(True, axis="y", alpha=0.3)
        _save_fig(out_dir / "feature_importances.png")

    try:
        import shap
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        plt.figure(figsize=(10, 7))
        shap.plots.bar(shap_values, show=False)
        _save_fig(out_dir / "shap_summary_bar.png")
        plt.figure(figsize=(10, 7))
        shap.plots.beeswarm(shap_values, show=False, max_display=20)
        _save_fig(out_dir / "shap_beeswarm.png")
    except Exception as e:
        with open(out_dir / "shap_info.txt", "w", encoding="utf-8") as f:
            f.write("SHAP недоступен: " + str(e))


def run_full_analysis(trades: List[Dict[str, Any]], out_dir: str | Path = "_viz_statistic") -> Path:
    out_dir = Path(out_dir); _ensure_dir(out_dir)
    df = flatten_trades(trades)
    df.to_csv(out_dir / "flattened_snap_params.csv", index=False, float_format="%.10g")
    build_stratified_heatmaps(df, out_dir / "strat_heatmaps")
    pc_df = compute_partial_correlations(df, out_dir / "partial_corr")
    pc_df.to_csv(out_dir / "partial_corr" / "partial_correlations.csv", float_format="%.6f")
    plot_ecdf_profit_by_signal(df, out_dir / "ecdf_profit_by_signal.png")
    plot_qq_profit_by_signal(df, out_dir / "qq_profit_by_signal.png")
    train_model_and_feature_importance(df, out_dir / "ml")
    return out_dir

def plot_profit_bars_with_stats(trades: list,
                                out_dir: str | Path = "_viz_statistic",
                                time_key_candidates: list | None = None,
                                figsize: tuple = (14, 7),
                                show_cumulative: bool = True,
                                dpi: int = 160,
                                max_ticks: int = 10) -> Path:
    """
    Столбиковая визуализация profit по сделкам, раскрашенная по type_of_signal,
    с подробной боковой панелью статистики справа (не перекрывает график),
    и компактной временной шкалой внизу (если есть open_time в ms или другие метки).

    Возвращает Path к сохранённому PNG.
    """
    from pathlib import Path
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import json
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from datetime import datetime

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if time_key_candidates is None:
        time_key_candidates = ["open_time", "time", "timestamp", "entry_time", "date"]

    # --- build dataframe ---
    rows = []
    for i, tr in enumerate(trades):
        profit = tr.get("profit", None)
        try:
            profit = float(profit) if profit is not None else np.nan
        except Exception:
            profit = np.nan
        sig = tr.get("type_of_signal", None)
        try:
            sig = int(sig) if sig is not None else None
        except Exception:
            pass
        ts_raw = None
        if isinstance(tr, dict) and "open_time" in tr:
            ts_raw = tr.get("open_time")
        else:
            for k in time_key_candidates:
                if isinstance(tr, dict) and k in tr:
                    ts_raw = tr.get(k)
                    break
        rows.append({"trade_idx": i, "profit": profit, "type_of_signal": sig, "_ts_raw": ts_raw})

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Список сделок пуст — нечего визуализировать.")

    # --- parse timestamps ---
    df["_timestamp"] = pd.NaT
    if "_ts_raw" in df.columns and df["_ts_raw"].notna().any():
        def try_parse_ts(v):
            try:
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    return pd.to_datetime(int(v), unit="ms", errors="coerce")
                if isinstance(v, str) and v.isdigit():
                    return pd.to_datetime(int(v), unit="ms", errors="coerce")
                return pd.to_datetime(v, errors="coerce")
            except Exception:
                return pd.NaT
        df["_timestamp"] = df["_ts_raw"].apply(try_parse_ts)

    use_time = df["_timestamp"].notna().sum() > 0
    if use_time:
        df = df.sort_values("_timestamp").reset_index(drop=True)
        x_label = "time"
    else:
        df = df.sort_values("trade_idx").reset_index(drop=True)
        x_label = "trade index"

    df["profit"] = pd.to_numeric(df["profit"], errors="coerce").fillna(0.0)
    df["type_of_signal"] = df["type_of_signal"].where(df["type_of_signal"].notna(), -1).astype(float)

    # --- drawdown helper ---
    def compute_max_drawdown(cum_array):
        res = {"max_dd_value": float("nan"), "max_dd_pct": float("nan"),
               "peak_idx": None, "trough_idx": None, "duration": None}
        if len(cum_array) == 0:
            return res
        cum = np.asarray(cum_array, dtype=float)
        running_max = np.maximum.accumulate(cum)
        drawdowns = running_max - cum
        max_dd = float(np.nanmax(drawdowns))
        if np.isnan(max_dd) or max_dd <= 0:
            return {"max_dd_value": 0.0, "max_dd_pct": 0.0, "peak_idx": None, "trough_idx": None, "duration": 0}
        trough_idx = int(np.nanargmax(drawdowns))
        peak_region = cum[:trough_idx + 1]
        if peak_region.size == 0:
            peak_idx = None
        else:
            peak_val = float(np.nanmax(peak_region))
            candidates = np.where(peak_region == peak_val)[0]
            peak_idx = int(candidates[-1]) if candidates.size > 0 else int(np.nanargmax(peak_region))
        peak_val = float(running_max[peak_idx]) if peak_idx is not None else float("nan")
        if peak_val and peak_val != 0:
            max_dd_pct = max_dd / abs(peak_val)
        else:
            max_dd_pct = float("nan")
        duration = None
        if peak_idx is not None:
            duration = trough_idx - peak_idx
        res.update({"max_dd_value": max_dd, "max_dd_pct": max_dd_pct,
                    "peak_idx": peak_idx, "trough_idx": trough_idx, "duration": duration})
        return res

    # --- overall / per-signal stats ---
    total_trades = int(len(df))
    num_wins = int((df["profit"] > 0).sum())
    num_losses = int((df["profit"] < 0).sum())
    sum_profit = float(df["profit"].sum())
    mean_profit = float(df["profit"].mean())
    median_profit = float(df["profit"].median())
    max_profit = float(df["profit"].max())
    max_loss = float(df["profit"].min())
    cum = np.cumsum(df["profit"].values)
    overall_draw = compute_max_drawdown(cum)
    overall_stats = {
        "total_trades": total_trades,
        "num_wins": num_wins,
        "num_losses": num_losses,
        "win_rate": num_wins / total_trades if total_trades > 0 else float("nan"),
        "sum_profit": sum_profit,
        "mean_profit": mean_profit,
        "median_profit": median_profit,
        "max_profit": max_profit,
        "max_loss": max_loss,
        "cumulative_final": float(cum[-1]) if len(cum) > 0 else 0.0,
        "max_drawdown": overall_draw
    }

    signal_values = sorted([int(s) for s in pd.unique(df["type_of_signal"]) if not (np.isnan(s))])
    per_signal_stats = {}
    for s in signal_values:
        df_s = df[df["type_of_signal"] == float(s)].copy().reset_index(drop=True)
        n = len(df_s)
        if n == 0:
            per_signal_stats[int(s)] = {}
            continue
        wins = int((df_s["profit"] > 0).sum())
        losses = int((df_s["profit"] < 0).sum())
        sum_p = float(df_s["profit"].sum())
        mean_p = float(df_s["profit"].mean())
        med_p = float(df_s["profit"].median())
        max_p = float(df_s["profit"].max())
        max_l = float(df_s["profit"].min())
        cum_s = np.cumsum(df_s["profit"].values)
        dd = compute_max_drawdown(cum_s)
        peak_time = None
        trough_time = None
        if dd["peak_idx"] is not None and use_time and dd["peak_idx"] < len(df_s):
            peak_time = str(pd.to_datetime(df_s.loc[dd["peak_idx"], "_timestamp"]))
        if dd["trough_idx"] is not None and use_time and dd["trough_idx"] < len(df_s):
            trough_time = str(pd.to_datetime(df_s.loc[dd["trough_idx"], "_timestamp"]))
        per_signal_stats[int(s)] = {
            "n": n,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / n if n > 0 else float("nan"),
            "sum_profit": sum_p,
            "mean_profit": mean_p,
            "median_profit": med_p,
            "max_profit": max_p,
            "max_loss": max_l,
            "cumulative_final": float(cum_s[-1]) if len(cum_s) > 0 else 0.0,
            "max_drawdown": dd,
            "peak_time": peak_time,
            "trough_time": trough_time
        }

    # --- plotting main axes + side panel axes ---
    palette = {1: "#1f77b4", 2: "#ff7f0e"}
    import itertools
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    extra_colors = itertools.cycle(default_colors)
    for s in signal_values:
        if int(s) not in palette:
            palette[int(s)] = next(extra_colors)
    bar_colors = [palette.get(int(sig), "#7f7f7f") for sig in df["type_of_signal"].astype(float).values]

    # create figure with wide layout, reserve right side for sidebar
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.05, 0.06, 0.70, 0.88])  # main axes: left, bottom, width, height (fractions)
    ax_side = fig.add_axes([0.78, 0.06, 0.20, 0.88])  # side panel for legend+stats
    ax_side.axis("off")

    x_pos = np.arange(len(df))
    bars = ax.bar(x_pos, df["profit"].values, color=bar_colors, edgecolor="black", linewidth=0.35)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel(x_label)
    ax.set_ylabel("profit")
    ax.set_title("Profit by trade — bars colored by type_of_signal")

    if show_cumulative:
        cum_series = np.cumsum(df["profit"].values)
        ax2 = ax.twinx()
        ax2.plot(x_pos, cum_series, color="black", linewidth=1.5, label="cumulative profit")
        ax2.set_ylabel("cumulative profit")
        if overall_draw["peak_idx"] is not None:
            pk = overall_draw["peak_idx"]
            tr_idx = overall_draw["trough_idx"]
            if pk is not None and pk < len(x_pos):
                ax2.plot(pk, cum_series[pk], marker="o", color="green", markersize=6)
            if tr_idx is not None and tr_idx < len(x_pos):
                ax2.plot(tr_idx, cum_series[tr_idx], marker="o", color="red", markersize=6)

    # --- compact X axis ticks (time-aware) ---
    if use_time:
        n = len(df)
        n_ticks = min(max_ticks, max(2, n))
        tick_idx = np.unique(np.linspace(0, n - 1, n_ticks, dtype=int))
        tick_pos = tick_idx
        times = pd.to_datetime(df.loc[tick_idx, "_timestamp"])
        span = pd.to_datetime(df["_timestamp"].iloc[-1]) - pd.to_datetime(df["_timestamp"].iloc[0]) if n > 1 else pd.Timedelta(0)
        if span >= pd.Timedelta(days=365):
            fmt = "%Y-%m"
        elif span >= pd.Timedelta(days=7):
            fmt = "%Y-%m-%d"
        elif span >= pd.Timedelta(days=1):
            fmt = "%Y-%m-%d\n%H:%M"
        else:
            fmt = "%H:%M\n%m-%d"
        tick_labels = [t.strftime(fmt) if not pd.isna(t) else "" for t in times]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    else:
        n = len(df)
        n_ticks = min(max_ticks, max(2, n))
        tick_idx = np.unique(np.linspace(0, n - 1, n_ticks, dtype=int))
        tick_labels = [str(int(df.loc[i, "trade_idx"])) for i in tick_idx]
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    # --- side panel: legend + stats text (no overlap) ---
    # build legend entries
    legend_handles = []
    for s in signal_values:
        legend_handles.append(Patch(facecolor=palette[int(s)], edgecolor='black', label=f"signal={int(s)}"))
    if show_cumulative:
        legend_handles.append(Line2D([0], [0], color='black', lw=1.5, label='cumulative profit'))

    # draw legend on side axis
    ax_side.legend(handles=legend_handles, loc="upper left", frameon=False)

    # prepare text lines for side axis
    def fmt_money(x):
        try:
            return f"{x:+.2f}"
        except Exception:
            return str(x)

    lines = []
    lines.append("SUMMARY:")
    lines.append(f"ALL: N={overall_stats['total_trades']}, wins={overall_stats['num_wins']}, losses={overall_stats['num_losses']}, win_rate={overall_stats['win_rate']:.1%}")
    lines.append(f"Sum: {fmt_money(overall_stats['sum_profit'])}, mean={overall_stats['mean_profit']:.3f}, median={overall_stats['median_profit']:.3f}")
    lines.append(f"Max profit: {fmt_money(overall_stats['max_profit'])}, Max loss: {fmt_money(overall_stats['max_loss'])}")
    dd = overall_stats["max_drawdown"]
    if dd and (not np.isnan(dd["max_dd_value"])):
        pk_time = None
        tr_time = None
        if use_time and dd["peak_idx"] is not None and dd["peak_idx"] < len(df):
            pk_time = str(pd.to_datetime(df.loc[dd["peak_idx"], "_timestamp"]))
        if use_time and dd["trough_idx"] is not None and dd["trough_idx"] < len(df):
            tr_time = str(pd.to_datetime(df.loc[dd["trough_idx"], "_timestamp"]))
        lines.append(f"Max drawdown: {dd['max_dd_value']:.2f} (pct {dd['max_dd_pct']:.2%})")
        lines.append(f"peak_idx={dd['peak_idx']}, trough_idx={dd['trough_idx']}, dur={dd['duration']}")
        if pk_time or tr_time:
            if pk_time: lines.append(f"peak_time={pk_time}")
            if tr_time: lines.append(f"trough_time={tr_time}")
    else:
        lines.append("Max drawdown: 0.00")

    lines.append("")  # spacer
    lines.append("BY SIGNAL:")
    for s in signal_values:
        st = per_signal_stats.get(int(s), {})
        if not st:
            continue
        lines.append(f"signal={s}: N={st['n']}, wins={st['wins']}, losses={st['losses']}, win_rate={st['win_rate']:.1%}")
        lines.append(f" sum={fmt_money(st['sum_profit'])}, mean={st['mean_profit']:.3f}, median={st['median_profit']:.3f}")
        lines.append(f" max={fmt_money(st['max_profit'])}, min={fmt_money(st['max_loss'])}, cum_final={fmt_money(st['cumulative_final'])}")
        dd_s = st["max_drawdown"]
        if dd_s and (not np.isnan(dd_s["max_dd_value"])):
            lines.append(f" max_dd={dd_s['max_dd_value']:.2f} (pct {dd_s['max_dd_pct']:.2%}), dur={dd_s['duration']}")
        lines.append("")

    # place text in ax_side using multiline text
    text = "\n".join(lines)
    ax_side.text(0.01, 0.02, text, fontsize=9, va="bottom", ha="left", wrap=True)

    # finalize & save
    save_file = out_path / "profit_bars_with_stats.png"
    plt.savefig(save_file, dpi=dpi)
    plt.close(fig)

    # save data + report
    save_csv = out_path / "profit_bars_plot_data.csv"
    save_json = out_path / "profit_bars_stats.json"
    df_to_save = df.copy()
    if "_timestamp" in df_to_save.columns:
        df_to_save["_timestamp"] = df_to_save["_timestamp"].astype(str)
    df_to_save.to_csv(save_csv, index=False, float_format="%.8g")
    report = {"overall": overall_stats, "per_signal": per_signal_stats, "created_at": datetime.utcnow().isoformat() + "Z",
              "n_rows_saved": int(len(df_to_save))}
    with open(save_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return save_file
