from datetime import datetime, timezone, timedelta
from statistics import mean, median
from typing import List, Dict, Any, Optional


def compute_trading_stats(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Подсчёт агрегированной статистики по списку сделок.

    Ожидаемый формат сделки (ключи):
      - open_time:  float | int  (мс от эпохи)
      - close_time: float | int  (мс от эпохи), опционально
      - profit:     float
      - duration:   float (в минутах), опционально (если нет — берём из close-open)
      - type_of_close: str
      - type_of_signal: int (1 или 2)

    Возвращает словарь с ключами:
      - overall:         агрегаты по всем сделкам
      - by_signal:       {1: {...}, 2: {...}} — те же агрегаты, но по каждому сигналу отдельно

    Внутри каждого блока:
      - n_trades
      - period: {start_iso, end_iso, span_days}
      - time_between_trades: {avg_seconds, avg_human, median_seconds, median_human}
      - trades_per_day: {avg_per_span, avg_per_active_day, active_days}
      - duration_stats: {avg_minutes, median_minutes}
      - drawdowns: {count, top10: [ {...} ]}  # эпизоды просадок (пункты и проценты)
      - type_of_close_stats: {close_type: {count, share, win_rate, sum_profit, mean_profit, median_profit}}
      - wins_losses: {wins, losses, zeros, win_rate, sum_profit}
    """
    # ----------------------- helpers -----------------------
    def to_dt(ms) -> Optional[datetime]:
        try:
            return datetime.fromtimestamp(float(ms) / 1000.0, tz=timezone.utc)
        except Exception:
            return None

    def fmt_human_seconds(sec: Optional[float]) -> Optional[str]:
        if sec is None:
            return None
        try:
            sec = float(sec)
        except Exception:
            return None
        if sec < 0:
            return f"{sec:.1f}s"
        d = int(sec // 86400)
        h = int((sec % 86400) // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        parts = []
        if d: parts.append(f"{d}d")
        if h or d: parts.append(f"{h}h")
        if m or h or d: parts.append(f"{m}m")
        parts.append(f"{s}s")
        return " ".join(parts)

    def safe_mean(seq: List[float]) -> Optional[float]:
        vals = [float(x) for x in seq if x is not None]
        return mean(vals) if vals else None

    def safe_median(seq: List[float]) -> Optional[float]:
        vals = [float(x) for x in seq if x is not None]
        return median(vals) if vals else None

    def compute_subset_stats(trades_subset: List[Dict[str, Any]]) -> Dict[str, Any]:
        # ----- basic vectors -----
        opens = [to_dt(t.get("open_time")) for t in trades_subset]
        closes = [to_dt(t.get("close_time")) for t in trades_subset]
        profits = [float(t.get("profit", 0.0) or 0.0) for t in trades_subset]
        durations = []
        for t, opn, cls in zip(trades_subset, opens, closes):
            dur = t.get("duration", None)
            if dur is None and opn and cls:
                dur = (cls - opn).total_seconds() / 60.0
            try:
                durations.append(float(dur) if dur is not None else None)
            except Exception:
                durations.append(None)
        type_of_close_vec = [t.get("type_of_close", None) for t in trades_subset]

        # ----- ordering by open_time -----
        idx_sorted = sorted(range(len(trades_subset)), key=lambda i: (opens[i] is None, opens[i]))
        opens = [opens[i] for i in idx_sorted]
        closes = [closes[i] for i in idx_sorted]
        profits = [profits[i] for i in idx_sorted]
        durations = [durations[i] for i in idx_sorted]
        type_of_close_vec = [type_of_close_vec[i] for i in idx_sorted]

        n = len(profits)

        # ----- period & trades per day -----
        start_dt = next((dt for dt in opens if dt is not None), None)
        end_dt = next((dt for dt in reversed(opens) if dt is not None), None)
        span_days = None
        if start_dt and end_dt and end_dt > start_dt:
            span_days = (end_dt - start_dt).total_seconds() / 86400.0

        # gaps between consecutive open times
        gaps_sec = []
        prev = None
        for dt in opens:
            if dt is not None and prev is not None:
                gaps_sec.append((dt - prev).total_seconds())
            prev = dt
        avg_gap = safe_mean(gaps_sec)
        med_gap = safe_median(gaps_sec)

        # trades per day (по всему интервалу) и по активным дням
        avg_per_span = (n / span_days) if (span_days and span_days > 0) else None
        # per active day
        active_day_counts = {}
        for dt in opens:
            if dt is None:
                continue
            key = dt.date()
            active_day_counts[key] = active_day_counts.get(key, 0) + 1
        active_days = len(active_day_counts)
        avg_per_active_day = (sum(active_day_counts.values()) / active_days) if active_days > 0 else None

        # ----- duration stats -----
        avg_dur = safe_mean(durations)
        med_dur = safe_median(durations)

        # ----- wins/losses -----
        wins = sum(1 for p in profits if p > 0)
        losses = sum(1 for p in profits if p < 0)
        zeros = n - wins - losses
        sum_profit = sum(profits)
        win_rate = wins / n if n > 0 else None

        # ----- type_of_close stats -----
        toc_stats = {}
        if n > 0:
            unique_toc = {}
            for toc, p in zip(type_of_close_vec, profits):
                if toc not in unique_toc:
                    unique_toc[toc] = []
                unique_toc[toc].append(p)
            for toc, plist in unique_toc.items():
                cnt = len(plist)
                wins_t = sum(1 for x in plist if x > 0)
                losses_t = sum(1 for x in plist if x < 0)
                toc_stats[toc] = {
                    "count": cnt,
                    "share": cnt / n,
                    "win_rate": wins_t / cnt if cnt > 0 else None,
                    "sum_profit": sum(plist),
                    "mean_profit": mean(plist) if cnt > 0 else None,
                    "median_profit": median(plist) if cnt > 0 else None,
                    "max_profit": max(plist) if cnt > 0 else None,
                    "max_loss": min(plist) if cnt > 0 else None,
                }

        # ----- drawdown episodes (cumulative by trade order) -----
        cum = []
        s = 0.0
        for p in profits:
            s += p
            cum.append(s)

        drawdowns = []
        if cum:
            peak_idx = 0
            peak_val = cum[0]
            trough_idx = 0
            trough_val = cum[0]
            in_dd = False

            for i in range(1, len(cum)):
                v = cum[i]
                if v >= peak_val:  # новая вершина или восстановление
                    if in_dd:
                        dd_value = peak_val - trough_val
                        denom = abs(peak_val) if abs(peak_val) > 1e-12 else None
                        dd_pct = (dd_value / denom) if denom is not None else None
                        start_t = opens[peak_idx].isoformat() if opens[peak_idx] else None
                        end_t = opens[trough_idx].isoformat() if opens[trough_idx] else None
                        dur_trades = trough_idx - peak_idx
                        dur_seconds = None
                        if opens[peak_idx] and opens[trough_idx]:
                            dur_seconds = (opens[trough_idx] - opens[peak_idx]).total_seconds()
                        drawdowns.append({
                            "peak_idx": peak_idx,
                            "trough_idx": trough_idx,
                            "start_time": start_t,
                            "end_time": end_t,
                            "dd_points": dd_value,
                            "dd_pct": dd_pct,
                            "duration_trades": dur_trades,
                            "duration_seconds": dur_seconds
                        })
                        in_dd = False
                    peak_idx = i
                    peak_val = v
                    trough_idx = i
                    trough_val = v
                else:  # v < peak_val -> просадка
                    if not in_dd:
                        in_dd = True
                        trough_idx = i
                        trough_val = v
                    else:
                        if v < trough_val:
                            trough_idx = i
                            trough_val = v

            # закрыть незакрытую просадку концом ряда
            if in_dd:
                dd_value = peak_val - trough_val
                denom = abs(peak_val) if abs(peak_val) > 1e-12 else None
                dd_pct = (dd_value / denom) if denom is not None else None
                start_t = opens[peak_idx].isoformat() if opens[peak_idx] else None
                end_t = opens[trough_idx].isoformat() if opens[trough_idx] else None
                dur_trades = trough_idx - peak_idx
                dur_seconds = None
                if opens[peak_idx] and opens[trough_idx]:
                    dur_seconds = (opens[trough_idx] - opens[peak_idx]).total_seconds()
                drawdowns.append({
                    "peak_idx": peak_idx,
                    "trough_idx": trough_idx,
                    "start_time": start_t,
                    "end_time": end_t,
                    "dd_points": dd_value,
                    "dd_pct": dd_pct,
                    "duration_trades": dur_trades,
                    "duration_seconds": dur_seconds
                })

        # сортируем и берём топ-10 по глубине в пунктах
        drawdowns_sorted = sorted(drawdowns, key=lambda d: d["dd_points"], reverse=True)
        top10 = drawdowns_sorted[:10]

        # собрать результат
        result = {
            "n_trades": n,
            "period": {
                "start_iso": opens[0].isoformat() if opens and opens[0] else None,
                "end_iso": opens[-1].isoformat() if opens and opens[-1] else None,
                "span_days": span_days
            },
            "time_between_trades": {
                "avg_seconds": avg_gap,
                "avg_human": fmt_human_seconds(avg_gap),
                "median_seconds": med_gap,
                "median_human": fmt_human_seconds(med_gap)
            },
            "trades_per_day": {
                "avg_per_span": avg_per_span,                 # N / календарный интервал (дни)
                "avg_per_active_day": avg_per_active_day,     # среднее по дням, где были сделки
                "active_days": active_days
            },
            "duration_stats": {
                "avg_minutes": avg_dur,
                "median_minutes": med_dur
            },
            "drawdowns": {
                "count": len(drawdowns_sorted),
                "top10": top10
            },
            "type_of_close_stats": toc_stats,
            "wins_losses": {
                "wins": wins,
                "losses": losses,
                "zeros": zeros,
                "win_rate": win_rate,
                "sum_profit": sum_profit
            }
        }
        return result

    # ----------------------- overall & by-signal -----------------------
    overall = compute_subset_stats(trades)

    # разбивка по сигналам
    by_signal: Dict[int, Dict[str, Any]] = {}
    for sig in (1, 2):
        sub = [t for t in trades if int(t.get("type_of_signal", 0) or 0) == sig]
        if sub:
            by_signal[sig] = compute_subset_stats(sub)
        else:
            by_signal[sig] = {
                "n_trades": 0,
                "period": {"start_iso": None, "end_iso": None, "span_days": None},
                "time_between_trades": {"avg_seconds": None, "avg_human": None, "median_seconds": None, "median_human": None},
                "trades_per_day": {"avg_per_span": None, "avg_per_active_day": None, "active_days": 0},
                "duration_stats": {"avg_minutes": None, "median_minutes": None},
                "drawdowns": {"count": 0, "top10": []},
                "type_of_close_stats": {},
                "wins_losses": {"wins": 0, "losses": 0, "zeros": 0, "win_rate": None, "sum_profit": 0.0},
            }

    return {
        "overall": overall,
        "by_signal": by_signal
    }



from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt


def plot_stats_overview(stats: dict,
                        out_dir: str | Path = "_viz_statistic",
                        filename: str = "overview_stats.png",
                        figsize: tuple = (16, 9),
                        dpi: int = 160) -> Path:
    """
    Комбинированная визуализация агрегированной статистики, возвращаемой compute_trading_stats().

    Вход:
      stats     — словарь, как в примере вывода compute_trading_stats(trades)
      out_dir   — папка для сохранения
      filename  — имя PNG-файла
      figsize   — размер фигуры
      dpi       — разрешение

    Вывод:
      Path к сохранённому изображению.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # --------- безопасные геттеры ---------
    def _get(d, *keys, default=None):
        for k in keys:
            if not isinstance(d, dict) or k not in d:
                return default
            d = d[k]
        return d

    ov = stats.get("overall", {})
    bysig = stats.get("by_signal", {})

    # --------- агрегаты Overall ---------
    n_trades = _get(ov, "n_trades", default=0) or 0
    span_days = _get(ov, "period", "span_days", default=None)
    start_iso = _get(ov, "period", "start_iso", default="-")
    end_iso = _get(ov, "period", "end_iso", default="-")

    avg_gap_s = _get(ov, "time_between_trades", "avg_seconds", default=None)
    med_gap_s = _get(ov, "time_between_trades", "median_seconds", default=None)

    avg_per_span = _get(ov, "trades_per_day", "avg_per_span", default=None)
    avg_per_active_day = _get(ov, "trades_per_day", "avg_per_active_day", default=None)
    active_days = _get(ov, "trades_per_day", "active_days", default=0) or 0

    avg_dur = _get(ov, "duration_stats", "avg_minutes", default=None)
    med_dur = _get(ov, "duration_stats", "median_minutes", default=None)

    wins = _get(ov, "wins_losses", "wins", default=0) or 0
    losses = _get(ov, "wins_losses", "losses", default=0) or 0
    zeros = _get(ov, "wins_losses", "zeros", default=0) or 0
    win_rate = _get(ov, "wins_losses", "win_rate", default=None)
    sum_profit = _get(ov, "wins_losses", "sum_profit", default=0.0) or 0.0
    mean_profit_overall = (sum_profit / n_trades) if n_trades else 0.0

    toc = _get(ov, "type_of_close_stats", default={}) or {}
    # подготовка данных по типам закрытия
    toc_labels = []
    toc_counts = []
    toc_means = []
    for k, v in toc.items():
        toc_labels.append(str(k))
        toc_counts.append(float(v.get("count", 0)))
        toc_means.append(float(v.get("mean_profit", 0.0)))

    # --------- ТОП-10 просадок ---------
    dd_list = _get(ov, "drawdowns", "top10", default=[]) or []
    # Оставляем топ-10 (уже отсортировано функцией compute_trading_stats)
    dd_points = [float(x.get("dd_points", 0.0)) for x in dd_list]
    dd_pct = [float(x.get("dd_pct", np.nan)) if x.get("dd_pct") is not None else np.nan for x in dd_list]
    dd_labels = []
    for x in dd_list:
        st = (x.get("start_time") or "").split("T")[0]
        en = (x.get("end_time") or "").split("T")[0]
        dur = x.get("duration_trades", None)
        dd_labels.append(f"{st}→{en}\n#{dur if dur is not None else '?'}")

    # --------- По сигналам ---------
    sigs_sorted = sorted([k for k in bysig.keys() if isinstance(k, int)])
    sig_sum_profit = []
    sig_win_rate = []
    sig_n = []
    sig_avg_dur = []
    for s in sigs_sorted:
        d = bysig.get(s, {})
        n = d.get("n_trades", 0) or 0
        wr = _get(d, "wins_losses", "win_rate", default=0.0) or 0.0
        sp = _get(d, "wins_losses", "sum_profit", default=0.0) or 0.0
        ad = _get(d, "duration_stats", "avg_minutes", default=np.nan)
        sig_sum_profit.append(sp)
        sig_win_rate.append(wr * 100.0)
        sig_n.append(n)
        sig_avg_dur.append(ad if ad is not None else np.nan)

    # --------- макет фигуры ---------
    fig = plt.figure(figsize=figsize)
    # основная сетка: 2 строки × 3 колонки, справа колонка шире под текст
    gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 1.2, 1.7], height_ratios=[1, 1], left=0.05, right=0.98, top=0.94, bottom=0.06, wspace=0.35, hspace=0.35)

    ax_toc = fig.add_subplot(gs[0, 0])     # распределение type_of_close
    ax_sig_sum = fig.add_subplot(gs[0, 1]) # суммарный профит по сигналам
    ax_dd = fig.add_subplot(gs[1, 0])      # ТОП-10 просадок
    ax_sig_dur = fig.add_subplot(gs[1, 1]) # средняя длительность по сигналам
    ax_side = fig.add_subplot(gs[:, 2])    # боковая панель текста
    ax_side.axis("off")

    # --------- (1) Бар-график: type_of_close (count) ---------
    if toc_labels:
        x = np.arange(len(toc_labels))
        ax_toc.bar(x, toc_counts, edgecolor="black", linewidth=0.6)
        ax_toc.set_xticks(x)
        ax_toc.set_xticklabels(toc_labels, rotation=20, ha="right")
        ax_toc.set_title("Типы закрытия: количество")
        ax_toc.set_ylabel("Count")
        # подпись среднего профита над столбиком
        for xi, c, m in zip(x, toc_counts, toc_means):
            ax_toc.text(xi, c, f"μ={m:+.2f}", ha="center", va="bottom", fontsize=9)
        ax_toc.grid(axis="y", alpha=0.25)
    else:
        ax_toc.text(0.5, 0.5, "Нет данных по type_of_close", ha="center", va="center")

    # --------- (2) Бар-график: суммарный профит по сигналам ---------
    if sigs_sorted:
        x = np.arange(len(sigs_sorted))
        bars = ax_sig_sum.bar(x, sig_sum_profit, edgecolor="black", linewidth=0.6)
        ax_sig_sum.set_xticks(x)
        ax_sig_sum.set_xticklabels([f"signal={s}" for s in sigs_sorted])
        ax_sig_sum.set_title("Суммарный profit по сигналам")
        ax_sig_sum.set_ylabel("Sum profit")
        ax_sig_sum.grid(axis="y", alpha=0.25)
        # подписи: win_rate и N над столбцами
        for xi, sp, wr, n in zip(x, sig_sum_profit, sig_win_rate, sig_n):
            ax_sig_sum.text(xi, sp + (0.02 * (max(sig_sum_profit) if sig_sum_profit else 1) + 1),
                            f"win={wr:.1f}%, N={n}", ha="center", va="bottom", fontsize=9)
    else:
        ax_sig_sum.text(0.5, 0.5, "Нет разбивки по сигналам", ha="center", va="center")

    # --------- (3) Гориз. бары: ТОП-10 просадок (в пунктах) ---------
    if dd_points:
        y = np.arange(len(dd_points))[::-1]  # сверху — крупнейшая
        ax_dd.barh(y, dd_points[::-1], edgecolor="black", linewidth=0.6)
        ax_dd.set_yticks(y)
        ax_dd.set_yticklabels(dd_labels[::-1], fontsize=8)
        ax_dd.set_xlabel("Drawdown (points)")
        ax_dd.set_title("ТОП-10 просадок (точки и %)")
        ax_dd.grid(axis="x", alpha=0.25)
        # подписи процентов справа от бара
        vals = dd_points[::-1]
        pcts = dd_pct[::-1]
        for yi, v, p in zip(y, vals, pcts):
            if not (p is None or np.isnan(p)):
                ax_dd.text(v * 1.01, yi, f"{p*100:.1f}%", va="center", ha="left", fontsize=8)
    else:
        ax_dd.text(0.5, 0.5, "Просадки отсутствуют", ha="center", va="center")

    # --------- (4) Средняя длительность по сигналам ---------
    if sigs_sorted and (not all(np.isnan(sig_avg_dur))):
        x = np.arange(len(sigs_sorted))
        ax_sig_dur.bar(x, sig_avg_dur, edgecolor="black", linewidth=0.6)
        ax_sig_dur.set_xticks(x)
        ax_sig_dur.set_xticklabels([f"signal={s}" for s in sigs_sorted])
        ax_sig_dur.set_title("Средняя длительность сделки по сигналам")
        ax_sig_dur.set_ylabel("Minutes")
        ax_sig_dur.grid(axis="y", alpha=0.25)
        for xi, v in zip(x, sig_avg_dur):
            if not (v is None or (isinstance(v, float) and math.isnan(v))):
                ax_sig_dur.text(xi, v, f"{v:.0f}m", ha="center", va="bottom", fontsize=9)
    else:
        ax_sig_dur.text(0.5, 0.5, "Нет данных по длительностям", ha="center", va="center")

    # --------- (5) Боковая панель — аккуратный текст ---------
    def _fmt_time_span(sec: float | None) -> str:
        if sec is None or (isinstance(sec, float) and math.isnan(sec)):
            return "NA"
        sec = float(sec)
        d = int(sec // 86400)
        h = int((sec % 86400) // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        parts = []
        if d: parts.append(f"{d}d")
        if h or d: parts.append(f"{h}h")
        if m or h or d: parts.append(f"{m}m")
        parts.append(f"{s}s")
        return " ".join(parts)

    # Общая сводка
    lines = []
    lines.append("SUMMARY")
    lines.append("───────")
    lines.append(f"N trades: {n_trades}")
    lines.append(f"Period: {start_iso} → {end_iso}")
    lines.append(f"Span (days): {span_days:.1f}" if isinstance(span_days, (int, float)) else "Span (days): NA")
    lines.append("")
    lines.append("Inter-trade time:")
    lines.append(f"  avg: {_fmt_time_span(avg_gap_s)}")
    lines.append(f"  median: {_fmt_time_span(med_gap_s)}")
    lines.append(f"Trades/day:")
    lines.append(f"  over span: {avg_per_span:.3f}" if isinstance(avg_per_span, (int, float)) else "  over span: NA")
    lines.append(f"  per active day: {avg_per_active_day:.3f}" if isinstance(avg_per_active_day, (int, float)) else "  per active day: NA")
    lines.append(f"  active days: {active_days}")
    lines.append("")
    lines.append("Duration:")
    lines.append(f"  avg: {avg_dur:.1f} min" if isinstance(avg_dur, (int, float)) else "  avg: NA")
    lines.append(f"  median: {med_dur:.1f} min" if isinstance(med_dur, (int, float)) else "  median: NA")
    lines.append("")
    lines.append("PnL:")
    lines.append(f"  wins: {wins} | losses: {losses} | zeros: {zeros}")
    lines.append(f"  win rate: {win_rate*100:.1f}%" if isinstance(win_rate, (int, float)) else "  win rate: NA")
    lines.append(f"  sum profit: {sum_profit:+.2f}")
    lines.append(f"  mean per trade: {mean_profit_overall:+.2f}")
    lines.append("")

    # Max drawdown (первый в списке)
    if dd_list:
        x0 = dd_list[0]
        pct0 = x0.get("dd_pct")
        pct_str = f"{pct0*100:.2f}%" if (pct0 is not None and not np.isnan(pct0)) else "NA"
        lines.append("Max drawdown:")
        lines.append(f"  {x0.get('dd_points', 0):.2f}  ({pct_str})")
        lines.append(f"  trades: {x0.get('duration_trades', 'NA')}")
        lines.append(f"  {x0.get('start_time','-')} → {x0.get('end_time','-')}")
        lines.append("")
    else:
        lines.append("Max drawdown: NA")
        lines.append("")

    # Сводка по типам закрытия
    lines.append("By type_of_close:")
    if toc_labels:
        for lbl, cnt, mu in zip(toc_labels, toc_counts, toc_means):
            share = cnt / n_trades if n_trades else 0.0
            lines.append(f"  {lbl}: N={int(cnt)}, share={share:.1%}, μ={mu:+.2f}")
    else:
        lines.append("  no data")
    lines.append("")

    # Сводка по сигналам
    lines.append("By signal:")
    if sigs_sorted:
        for s, n, wr, sp in zip(sigs_sorted, sig_n, sig_win_rate, sig_sum_profit):
            mean_s = (sp / n) if n else 0.0
            lines.append(f"  signal={s}: N={n}, win={wr:.1f}%, sum={sp:+.2f}, μ={mean_s:+.2f}")
    else:
        lines.append("  no data")

    # Печать в правую панель
    ax_side.text(0.02, 0.98, "\n".join(lines), fontsize=10, va="top", ha="left")

    fig.suptitle("Trading Overview", fontsize=14, y=0.995)
    save_path = out_path / filename
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    return save_path
