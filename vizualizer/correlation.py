# analyze_viz.py (dynamic metrics, robust numeric & signals)
# ============================================================================
# Гибкий анализ и визуализация для произвольного набора метрик из metrics.
# Главное: редактируйте METRIC_KEYS ниже — весь отчёт подстроится.
# Устойчив к "сложным" type_of_signal (dict/str/None) и к странным типам в метриках.
# Добавлено: metrics_summary.csv (nunique/min/max/… для контроля данных).
# ============================================================================

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shared_vars as sv
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ============================== НАСТРОЙКИ ====================================
# >>> РЕДАКТИРУЙТЕ ЭТОТ СПИСОК <<<
# Укажите ключи из metrics, которые хотите анализировать прямо сейчас.
METRIC_KEYS: List[str] = [
    
]

# Предпочтительная тройка для 3D (если каких-то метрик нет, модуль подберёт доступные)
TRIPLE_FOR_3D: List[str] = ["premium_pct_spot", "hours_to_expiry", "iv"]

# Метрика для размера маркера в 3D (если присутствует среди выбранных)
SIZE_METRIC_FOR_3D: Optional[str] = "atr_rel"

# Параметры бининга для 2D-теплокарт
DEFAULT_BINS_X = 10
DEFAULT_BINS_Y = 10
DEFAULT_BINNING = "quantile"   # "quantile" | "uniform"
DEFAULT_MIN_COUNT = 10


# ============================== УТИЛИТЫ ======================================

def _safe_float(x) -> float | None:
    """Пытается привести значение к float; NaN/inf/ошибка -> None (чтобы потом отфильтровать)."""
    try:
        if x is None:
            return None
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _sanitize_signal(sig: Any) -> Tuple[str, Optional[float]]:
    """
    Безопасное представление сигнала:
      - cat: строковая метка (для легенд/группировок),
      - num: числовое представление, если его можно однозначно получить.
    Случай dict: пробуем один из ключей ('id','value','signal','code','type').
    """
    if isinstance(sig, dict):
        for key in ("id", "value", "signal", "code", "type"):
            if key in sig:
                try:
                    num = float(sig[key])
                    return (f"{key}={sig[key]}", num if math.isfinite(num) else None)
                except Exception:
                    return (f"{key}={sig[key]}", None)
        return (str(sig), None)
    try:
        num = float(sig)
        return (str(sig), num if math.isfinite(num) else None)
    except Exception:
        return (str(sig), None)


def _flatten(tr: Dict[str, Any], metric_keys: List[str]) -> Dict[str, Any]:
    """Плоская строка: базовые поля + только нужные метрики из metrics."""
    cm = tr.get("metrics", {}) or {}
    cat, num = _sanitize_signal(tr.get("type_of_signal"))
    row: Dict[str, Any] = {
        "open_time_ms": _safe_float(tr.get("open_time")),
        "close_time_ms": _safe_float(tr.get("close_time")),
        "type_of_signal_cat": cat,
        "type_of_signal_num": num,
        "profit": _safe_float(tr.get("profit")),
    }
    for k in metric_keys:
        row[k] = _safe_float(cm.get(k))
    return row


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _present_metrics(df: pd.DataFrame, metric_keys: List[str]) -> List[str]:
    """Метрики из списка, которые реально есть и имеют хотя бы одно валидное число."""
    out = []
    for k in metric_keys:
        if k in df.columns and pd.to_numeric(df[k], errors="coerce").notna().sum() > 0:
            out.append(k)
    return out


def _to_numeric_col(df: pd.DataFrame, col: str) -> pd.Series:
    """Надёжно приводит столбец к числу (возвращает Series float с NaN на мусоре)."""
    return pd.to_numeric(df[col], errors="coerce")


def _bin_2d(
    x: pd.Series,
    y: pd.Series,
    values: pd.Series,
    n_bins_x: int = DEFAULT_BINS_X,
    n_bins_y: int = DEFAULT_BINS_Y,
    binning: str = DEFAULT_BINNING,
    min_count: int = DEFAULT_MIN_COUNT,
    agg: str = "mean",  # "mean" | "share_pos"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """2D-биннинг и агрегирование. Возвращает (grid, x_edges, y_edges, counts)."""
    data = pd.DataFrame({"x": x, "y": y, "v": values}).dropna()
    if data.empty:
        return (np.full((n_bins_y, n_bins_x), np.nan), np.array([]), np.array([]), np.zeros((n_bins_y, n_bins_x)))

    if binning == "quantile":
        try:
            x_bins = pd.qcut(data["x"], q=n_bins_x, duplicates="drop")
            y_bins = pd.qcut(data["y"], q=n_bins_y, duplicates="drop")
        except Exception:
            x_bins = pd.cut(data["x"], bins=n_bins_x)
            y_bins = pd.cut(data["y"], bins=n_bins_y)
    else:
        x_bins = pd.cut(data["x"], bins=n_bins_x)
        y_bins = pd.cut(data["y"], bins=n_bins_y)

    grouped = data.groupby([y_bins, x_bins])
    counts = grouped.size().unstack(fill_value=0)

    if agg == "mean":
        grid = grouped["v"].mean().unstack()
    elif agg == "share_pos":
        grid = grouped["v"].apply(lambda s: (s > 0).mean()).unstack()
    else:
        raise ValueError("agg must be 'mean' or 'share_pos'")

    grid = grid.where(counts >= min_count)

    def _edges_from_index(ii: pd.IntervalIndex) -> np.ndarray:
        if ii is None or len(ii) == 0:
            return np.array([])
        lefts = ii.left.to_numpy(dtype=float)
        rights = ii.right.to_numpy(dtype=float)
        return np.r_[lefts[0], rights]

    x_edges = _edges_from_index(grid.columns.categories if hasattr(grid.columns, "categories") else grid.columns)
    y_edges = _edges_from_index(grid.index.categories if hasattr(grid.index, "categories") else grid.index)

    return grid.to_numpy(), x_edges, y_edges, counts.to_numpy()


def _plot_heatmap2d(
    grid: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray,
    title: str, xlabel: str, ylabel: str, out_path: Path
) -> None:
    if grid.size == 0:
        return
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(grid, origin="lower", aspect="auto")
    if len(x_edges) > 1:
        ax.set_xticks(np.arange(grid.shape[1]))
        ax.set_xticklabels([f"{i+1}" for i in range(grid.shape[1])])
    if len(y_edges) > 1:
        ax.set_yticks(np.arange(grid.shape[0]))
        ax.set_yticklabels([f"{i+1}" for i in range(grid.shape[0])])
    ax.set_title(title)
    ax.set_xlabel(xlabel + " (bins)")
    ax.set_ylabel(ylabel + " (bins)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ============================== БАЗОВЫЙ ОТЧЁТ ===============================

def analyze_trades(
    trades: List[Dict[str, Any]],
    out_dir: str | Path = "_viz_statistic",
    metric_keys: Optional[List[str]] = None,  # если None — берём METRIC_KEYS
) -> Dict[str, Any]:
    """
    Корреляции выбранных метрик с profit + scatter по безопасной категории type_of_signal_cat.
    """
    out_path = Path(out_dir)
    _ensure_dir(out_path)

    keys = metric_keys or METRIC_KEYS
    if not keys:
        raise ValueError("METRIC_KEYS пуст — укажите хотя бы одну метрику.")
    rows = [_flatten(t, keys) for t in trades]
    df = pd.DataFrame(rows)

    # Приведение времени (на будущее)
    for col in ("open_time_ms", "close_time_ms"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time_ms"], unit="ms", errors="coerce", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time_ms"], unit="ms", errors="coerce", utc=True)

    # Безопасная категория сигнала
    df["type_of_signal_cat"] = df["type_of_signal_cat"].astype(str)

    # Сохраняем плоские данные
    df.to_csv(out_path / "data_flat.csv", index=False)

    # Диагностика метрик (чтобы отлавливать "все точки в одну линию")
    metrics_all = _present_metrics(df, keys)
    diag_rows = []
    for m in metrics_all + ["profit"]:
        s = _to_numeric_col(df, m)
        diag_rows.append({
            "metric": m,
            "n": int(s.notna().sum()),
            "nunique": int(s.dropna().nunique()),
            "min": float(s.min()) if s.notna().any() else np.nan,
            "max": float(s.max()) if s.notna().any() else np.nan,
            "mean": float(s.mean()) if s.notna().any() else np.nan,
            "std": float(s.std(ddof=1)) if s.notna().sum() > 1 else 0.0,
        })
    pd.DataFrame(diag_rows).set_index("metric").to_csv(out_path / "metrics_summary.csv")

    if not metrics_all:
        raise ValueError("В данных нет ни одной валидной числовой метрики из METRIC_KEYS.")

    # Корреляции с profit
    corr_rows = []
    for m in metrics_all:
        x = _to_numeric_col(df, m)
        y = _to_numeric_col(df, "profit")
        mask = x.notna() & y.notna()
        if mask.sum() < 2:
            pear = spear = np.nan
        else:
            sub = pd.DataFrame({m: x[mask], "profit": y[mask]})
            pear = float(sub.corr(method="pearson").iloc[0, 1])
            spear = float(sub.corr(method="spearman").iloc[0, 1])
        corr_rows.append({"metric": m, "pearson": pear, "spearman": spear})
    pd.DataFrame(corr_rows).set_index("metric").to_csv(out_path / "corr_vs_profit.csv")

    # Scatter: по каждой метрике vs profit с разрезом по сигналам
    markers = ["o", "^", "s", "X", "D", "P", "v", "<", ">", "*"]
    classes = [str(x) for x in pd.Index(df["type_of_signal_cat"].unique().tolist())]
    classes = classes or ["all"]
    sig_to_marker = {c: markers[i % len(markers)] for i, c in enumerate(classes)}

    for m in metrics_all:
        x = _to_numeric_col(df, m)
        y = _to_numeric_col(df, "profit")
        cat = df["type_of_signal_cat"].astype(str)
        mask = x.notna() & y.notna()
        if mask.sum() == 0:
            continue

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111)
        for c in sorted(pd.Index(cat[mask].unique().tolist())):
            submask = mask & (cat == c)
            if submask.sum() == 0:
                continue
            ax.scatter(x[submask].to_numpy(dtype=float),
                       y[submask].to_numpy(dtype=float),
                       marker=sig_to_marker.get(c, "o"),
                       alpha=0.85, label=f"signal={c}")
        ax.set_xlabel(m); ax.set_ylabel("profit")
        ax.set_title(f"{m} vs profit (по type_of_signal)")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_path / f"scatter_{m}_vs_profit_by_signal.png", dpi=150)
        plt.close(fig)

    summary = {
        "out_dir": str(out_path),
        "n_rows": int(df.shape[0]),
        "metrics_used": metrics_all,
        "corr_csv": str(out_path / "corr_vs_profit.csv"),
        "metrics_summary_csv": str(out_path / "metrics_summary.csv"),
        "scatter_images": [str(out_path / f"scatter_{m}_vs_profit_by_signal.png") for m in metrics_all],
    }
    with open(out_path / "result_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    return summary


# ====================== СОВМЕСТНЫЕ ВИЗУАЛИЗАЦИИ (ПАРЫ/3D) ===================

def visualize_joint(
    trades: List[Dict[str, Any]],
    out_dir: str | Path = "_viz_statistic",
    metric_keys: Optional[List[str]] = None,
    n_bins_x: int = DEFAULT_BINS_X,
    n_bins_y: int = DEFAULT_BINS_Y,
    binning: str = DEFAULT_BINNING,
    min_count: int = DEFAULT_MIN_COUNT,
    make_3d: bool = True,
    triple_for_3d: Optional[List[str]] = None,
    size_metric_for_3d: Optional[str] = SIZE_METRIC_FOR_3D,
) -> Dict[str, Any]:
    """
    2D-теплокарты mean(profit)/share(profit>0) для всех пар выбранных метрик +
    (опц.) 3D-scatter по тройке метрик (цвет=profit, размер∝size_metric_for_3d).
    """
    out_path = Path(out_dir)
    _ensure_dir(out_path)

    keys = metric_keys or METRIC_KEYS
    if not keys:
        raise ValueError("METRIC_KEYS пуст — укажите хотя бы одну метрику.")
    rows = [_flatten(t, keys) for t in trades]
    df = pd.DataFrame(rows)

    metrics = _present_metrics(df, keys)
    if not metrics:
        raise ValueError("В данных нет ни одной валидной числовой метрики из METRIC_KEYS.")

    # Все пары
    pairs: List[Tuple[str, str]] = []
    for i in range(len(metrics)):
        for j in range(i + 1, len(metrics)):
            pairs.append((metrics[i], metrics[j]))

    heatmap_files: List[str] = []
    for x_col, y_col in pairs:
        x = _to_numeric_col(df, x_col)
        y = _to_numeric_col(df, y_col)
        v = _to_numeric_col(df, "profit")
        sub = pd.DataFrame({x_col: x, y_col: y, "profit": v}).dropna()
        if sub.empty:
            continue

        # mean(profit)
        grid_mean, x_edges, y_edges, _ = _bin_2d(
            sub[x_col], sub[y_col], sub["profit"],
            n_bins_x=n_bins_x, n_bins_y=n_bins_y,
            binning=binning, min_count=min_count, agg="mean"
        )
        mean_csv = out_path / f"heatmap_mean_profit_{x_col}_x_{y_col}.csv"
        mean_png = out_path / f"heatmap_mean_profit_{x_col}_x_{y_col}.png"
        pd.DataFrame(grid_mean).to_csv(mean_csv, index=False)
        _plot_heatmap2d(grid_mean, x_edges, y_edges, f"mean(profit): {x_col} × {y_col}", x_col, y_col, mean_png)
        heatmap_files.append(str(mean_png))

        # share(profit>0)
        grid_share, x_edges, y_edges, _ = _bin_2d(
            sub[x_col], sub[y_col], sub["profit"],
            n_bins_x=n_bins_x, n_bins_y=n_bins_y,
            binning=binning, min_count=min_count, agg="share_pos"
        )
        share_csv = out_path / f"heatmap_share_pos_{x_col}_x_{y_col}.csv"
        share_png = out_path / f"heatmap_share_pos_{x_col}_x_{y_col}.png"
        pd.DataFrame(grid_share).to_csv(share_csv, index=False)
        _plot_heatmap2d(grid_share, x_edges, y_edges, f"share(profit>0): {x_col} × {y_col}", x_col, y_col, share_png)
        heatmap_files.append(str(share_png))

    # 3D-scatter: автоматически подберём доступную тройку
    scatter3d_file = None
    if make_3d and len(metrics) >= 3:
        triple_pref = triple_for_3d or TRIPLE_FOR_3D
        triple = [m for m in triple_pref if m in metrics]
        for m in metrics:
            if len(triple) >= 3:
                break
            if m not in triple:
                triple.append(m)

        if len(triple) >= 3:
            cols_needed = triple[:3] + ["profit"]
            size_key = size_metric_for_3d if (size_metric_for_3d in metrics) else None
            if size_key:
                cols_needed.append(size_key)

            sub = df[cols_needed].copy()
            for c in cols_needed:
                sub[c] = _to_numeric_col(sub, c)
            sub = sub.dropna()
            if not sub.empty:
                if size_key:
                    sdata = sub[size_key].to_numpy(dtype=float)
                    if np.nanmax(sdata) > np.nanmin(sdata):
                        sizes = 20 + 180 * (sdata - np.nanmin(sdata)) / (np.nanmax(sdata) - np.nanmin(sdata))
                    else:
                        sizes = np.full_like(sdata, 40.0)
                else:
                    sizes = 40.0

                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection="3d")
                sc = ax.scatter(
                    sub[triple[0]].to_numpy(dtype=float),
                    sub[triple[1]].to_numpy(dtype=float),
                    sub[triple[2]].to_numpy(dtype=float),
                    c=sub["profit"].to_numpy(dtype=float),
                    s=sizes,
                    depthshade=True,
                )
                ax.set_xlabel(triple[0]); ax.set_ylabel(triple[1]); ax.set_zlabel(triple[2])
                title = f"3D: ({triple[0]}, {triple[1]}, {triple[2]}) — цвет=profit"
                if size_key: title += f", размер∝{size_key}"
                ax.set_title(title)
                fig.colorbar(sc, ax=ax)
                fig.tight_layout()
                scatter3d_file = out_path / f"scatter3d_{triple[0]}_{triple[1]}_{triple[2]}_color_profit.png"
                fig.savefig(scatter3d_file, dpi=150)
                plt.close(fig)

    result = {
        "out_dir": str(out_path),
        "metrics_used": metrics,
        "heatmaps": heatmap_files,
        "scatter3d": str(scatter3d_file) if scatter3d_file else None,
    }
    with open(out_path / "joint_viz_summary.json", "w", encoding="utf-8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=2)
    return result

