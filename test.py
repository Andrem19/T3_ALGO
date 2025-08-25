from data.load_cand import load_klines_numpy, load_enriched_klines
from datetime import datetime
import time
import shared_vars as sv
import copy
import vizualizer.viz as viz
import statistic.stat as stat
import vizualizer.correlation as cor

import math

# ------- утилиты -------

def _clip(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x)) if x is not None else None

def _ret_oc(bar):
    r = bar.get("ret_oc")
    if r is None:
        o, c = bar.get("open"), bar.get("close")
        if o and o != 0 and c is not None:
            r = (c - o) / o
    return r

def dominance_from_bar(bar, side="perp"):
    """Нормированная доминанта: (buy - sell) / (buy + sell) ∈ [-1..1]."""
    b = bar.get(f"flows_{side}_buy_q")
    s = bar.get(f"flows_{side}_sell_q")
    if b is None or s is None:
        return None
    tot = b + s
    if tot <= 0:
        return 0.0
    return (b - s) / tot

def trades_delta(prev_bar, bar):
    """Относительное изменение числа сделок ∈ [-1..1] (клип)."""
    if prev_bar is None: 
        return None
    t0, t1 = prev_bar.get("trades"), bar.get("trades")
    if t0 is None or t1 is None or t0 <= 0:
        return None
    return _clip((t1 - t0) / t0, -1.0, 1.0)

# ------- индикаторы -------

def flow_sync_index(bar):
    """
    Синхронность спота и перпа: product(dom_spot, dom_perp) ∈ [-1..1].
    >0 — обе стороны в одну сторону; <0 — расходятся.
    """
    ds = dominance_from_bar(bar, "spot")
    dp = dominance_from_bar(bar, "perp")
    if ds is None or dp is None:
        return None
    return ds * dp

def flow_divergence_index(bar):
    """
    Перп против спота: (dom_perp - dom_spot)/2 ∈ [-1..1].
    >0 — перп бычей относительно спота; <0 — медвежий.
    """
    ds = dominance_from_bar(bar, "spot")
    dp = dominance_from_bar(bar, "perp")
    if ds is None or dp is None:
        return None
    return _clip((dp - ds) / 2.0)

def flow_price_confluence(bar, side="perp", r_scale=0.01):
    """
    Согласованность потока и цены: dom * sign(ret) * min(|ret|/r_scale,1).
    ∈ [-1..1]. >0 — поток поддержал ход свечи, <0 — против хода.
    """
    dom = dominance_from_bar(bar, side)
    r = _ret_oc(bar)
    if dom is None or r is None:
        return None
    mag = _clip(abs(r) / r_scale)
    sgn = 1.0 if r > 0 else -1.0 if r < 0 else 0.0
    return dom * sgn * mag

def flow_trades_confirmation(prev_bar, bar, side="perp", t_cap=1.0):
    """
    Подтверждение потока всплеском сделок: dom * Δtrades.
    Δtrades = clip((trades_t - trades_{t-1})/trades_{t-1}, ±t_cap) ∈ [-1..1].
    """
    dom = dominance_from_bar(bar, side)
    dt = trades_delta(prev_bar, bar)
    if dom is None or dt is None:
        return None
    return _clip(dom * dt)

def absorption_signal(bar, side="perp", r_scale=0.01):
    """
    'Абсорбция' (поток против результата свечи):
    = - flow_price_confluence  (т.е. >0 — признаки абсорбции)
    """
    v = flow_price_confluence(bar, side=side, r_scale=r_scale)
    return None if v is None else -v

def flow_momentum(prev_bar, bar, side="perp"):
    """
    Моментум доминанты: dom_t - dom_{t-1} ∈ [-2..2], клип к [-1..1].
    >0 — сдвиг в сторону покупателей, <0 — к продавцам.
    """
    d1 = dominance_from_bar(bar, side)
    d0 = dominance_from_bar(prev_bar, side) if prev_bar else None
    if d1 is None or d0 is None:
        return None
    return _clip(d1 - d0)

def taker_activity_share(bar, side="perp"):
    """
    Доля 'активности' тэйкеров: (buy+sell)/quote_volume, клип к [0..1.5]→[0..1].
    >0.66 — бар с доминированием тэйкер-торговли.
    """
    b = bar.get(f"flows_{side}_buy_q"); s = bar.get(f"flows_{side}_sell_q")
    qv = bar.get("quote_volume")
    if b is None or s is None or qv is None or qv <= 0:
        return None
    return _clip((b + s) / qv, 0.0, 1.0)

def triple_confluence(prev_bar, bar):
    """
    Композит из 3 факторов (перп-доминанта, знак свечи, всплеск сделок):
    = flow_price_confluence(perp) * (0.5 + 0.5*max(0, Δtrades))
    Усиливает сигнал, когда сделки растут.
    """
    base = flow_price_confluence(bar, side="perp", r_scale=0.01)
    dt = trades_delta(prev_bar, bar)
    if base is None or dt is None:
        return None
    boost = 0.5 + 0.5 * max(0.0, dt)  # 0.5..1.0
    return _clip(base * boost)

def spot_perp_agreement_strength(bar):
    """
    Сила согласия рынков: 0.5*(|dom_spot|+|dom_perp|) * sign(dom_spot*dom_perp) ∈ [-1..1].
    """
    ds = dominance_from_bar(bar, "spot")
    dp = dominance_from_bar(bar, "perp")
    if ds is None or dp is None:
        return None
    sign = 1.0 if ds*dp > 0 else -1.0 if ds*dp < 0 else 0.0
    return _clip(0.5 * (abs(ds) + abs(dp)) * sign)




def signed_dominance(buy_q, sell_q):
    """(buy - sell) / (buy + sell). Возвращает None, если данных нет."""
    if buy_q is None or sell_q is None:
        return None
    total = buy_q + sell_q
    if total <= 0:
        return 0.0
    return (buy_q - sell_q) / total

# spot_path = '/home/jupiter/PYTHON/MARKET_DATA/_crypto_data/spot_BTCUSDT/1h/BTCUSDT-1h.csv' 
fut_path = '/home/jupiter/PYTHON/MARKET_DATA/_crypto_data/fut_BTCUSDT/1h/BTCUSDT-1h.csv'

if __name__ == "__main__": 
    start = datetime(2020, 1, 1) 
    end = datetime(2025, 1, 1)
    
    # btc_spot = load_enriched_klines(spot_path, start, end)
    btc_fut = load_enriched_klines(fut_path, start, end)
    for i in range(2, len(btc_fut)-10):
        profit = (btc_fut[i+1]['close'] - btc_fut[i+1]['open']) *0.01
        spot_dom = signed_dominance(btc_fut[i].get("flows_spot_buy_q"), btc_fut[i].get("flows_spot_sell_q"))
        perp_dom = signed_dominance(btc_fut[i].get("flows_perp_buy_q"), btc_fut[i].get("flows_perp_sell_q"))  
        trades_direction = btc_fut[i]['trades']-btc_fut[i-1]['trades']
        
        prev_bar = btc_fut[i-1]; bar = btc_fut[i]
        sv.metrics['dom_spot']  = dominance_from_bar(bar, "spot")
        sv.metrics['dom_perp']  = dominance_from_bar(bar, "perp")
        sv.metrics['flow_sync'] = flow_sync_index(bar)
        sv.metrics['flow_div']  = flow_divergence_index(bar)
        sv.metrics['conf_perp'] = flow_price_confluence(bar, side="perp", r_scale=0.01)
        sv.metrics['conf_spot'] = flow_price_confluence(bar, side="spot", r_scale=0.01)
        sv.metrics['absorb_p']  = absorption_signal(bar, side="perp", r_scale=0.01)
        sv.metrics['dt_trades'] = trades_delta(prev_bar, bar)
        sv.metrics['conf_tr_p'] = flow_trades_confirmation(prev_bar, bar, side="perp")
        sv.metrics['dom_mom_p'] = flow_momentum(prev_bar, bar, side="perp")
        sv.metrics['taker_share_p'] = taker_activity_share(bar, side="perp")
        sv.metrics['triple']    = triple_confluence(prev_bar, bar)
        sv.metrics['agree_sp']  = spot_perp_agreement_strength(bar)

        report = {
            'open_time': btc_fut[i+1]['open_time'],
            'close_time': btc_fut[i+1]['close_time'],
            'type_of_signal': 1,
            'type_of_close': 'close',
            'profit': profit - (0.01* btc_fut[i+1]['open'] * 0.00045 * 2),
            'duration': 60,
            'metrics': copy.deepcopy(sv.metrics)
        }
        sv.positions_list.append(report)
    viz.plot_profit_bars_with_stats(sv.positions_list, out_dir="_viz_statistic")
    st = stat.compute_trading_stats(sv.positions_list)
    stat.plot_stats_overview(st, out_dir="_viz_statistic", filename="overview_stats.png")
    cor.analyze_trades(sv.positions_list, metric_keys=list(sv.metrics.keys()))
    

# -*- coding: utf-8 -*-
# """
# Обогащение исторических CSV (klines) индикаторами:
# - price: ret_oc, ret_cc
# - flows: spot & perp (из taker_buy_quote и quote_volume)
# - stablecoins: USDC/FDUSD/USDP (цена и отклонение от 1.0)
# - breadth (частично): up/down/flat/advance_decline по споту USDT
# - TA: RSI(14), ATR(14)

# Перезаписывает исходные файлы с хедерами и добавленными колонками.
# """

# import os
# import glob
# import math
# import numpy as np
# import pandas as pd

# # TA-Lib (должен быть установлен)
# import talib as ta

# BASE_DIR = "/home/jupiter/PYTHON/MARKET_DATA/_crypto_data"
# TIMEFRAMES = ["1m","5m","15m","30m","1h","4h","1d"]

# # --- именованные колонки kline ---
# KCOLS = [
#     "open_time","open","high","low","close","volume",
#     "close_time","quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"
# ]

# STABLE_SYMBOLS = {"USDCUSDT","FDUSDUSDT","USDPUSDT","BUSDUSDT","TUSDUSDT"}  # расширено на всякий
# # Короткие имена для колонок стейблов
# STABLE_MAP = {
#     "USDCUSDT": ("usdc_px","usdc_dev"),
#     "FDUSDUSDT": ("fdusd_px","fdusd_dev"),
#     "USDPUSDT": ("usdp_px","usdp_dev"),
# }

# # ---------- Утилиты чтения/преобразования ----------
# def read_klines_csv(path: str) -> pd.DataFrame:
#     """Читает CSV без хедера, чистит мусор, приводит типы, сортирует по open_time."""
#     df = pd.read_csv(path, header=None)
#     # убрать потенциальные строковые заголовки / мусор
#     df = df[pd.to_numeric(df[0], errors="coerce").notnull()].copy()
#     df.columns = KCOLS
#     # приведение типов
#     for c in ["open_time","close_time","trades"]:
#         df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
#     for c in ["open","high","low","close","volume","quote_volume","taker_buy_base","taker_buy_quote","ignore"]:
#         df[c] = pd.to_numeric(df[c], errors="coerce")
#     # сортировка и избавление от дубликатов таймстампов
#     df = df.sort_values("open_time")
#     df = df.groupby("open_time", as_index=False).last()
#     return df

# def write_csv_overwrite(df: pd.DataFrame, path: str):
#     """Перезаписывает CSV c header'ом."""
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     tmp = f"{path}.tmp"
#     df.to_csv(tmp, index=False)
#     os.replace(tmp, path)

# def compute_price_ta(df: pd.DataFrame) -> pd.DataFrame:
#     """Добавляет ret_oc, ret_cc, rsi14, atr14."""
#     out = df.copy()
#     # доходности
#     out["ret_oc"] = (out["close"] - out["open"]) / out["open"]
#     out["ret_cc"] = out["close"].pct_change()
#     # TA-Lib
#     close = out["close"].astype(float).values
#     high  = out["high"].astype(float).values
#     low   = out["low"].astype(float).values
#     out["rsi14"] = ta.RSI(close, timeperiod=14)
#     out["atr14"] = ta.ATR(high, low, close, timeperiod=14)
#     return out

# def compute_flows_cols(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
#     """flows_{prefix}_buy_q/sell_q/net_q из kline."""
#     out = pd.DataFrame({"open_time": df["open_time"].values})
#     buy = df["taker_buy_quote"].astype(float)
#     sell = df["quote_volume"].astype(float) - df["taker_buy_quote"].astype(float)
#     out[f"flows_{prefix}_buy_q"] = buy.values
#     out[f"flows_{prefix}_sell_q"] = sell.values
#     out[f"flows_{prefix}_net_q"] = (buy - sell).values
#     return out

# def load_stable_pack(tf: str) -> pd.DataFrame:
#     """
#     Готовит DataFrame со столбцами стейблов, соединённых по open_time:
#     usdc_px/usdc_dev, fdusd_px/fdusd_dev, usdp_px/usdp_dev
#     """
#     frames = []
#     for sym, (px_col, dev_col) in STABLE_MAP.items():
#         path = f"{BASE_DIR}/spot_{sym}/{tf}/{sym}-{tf}.csv"
#         if not os.path.exists(path):
#             continue
#         df = read_klines_csv(path)[["open_time","close"]].rename(columns={"close": px_col})
#         df[dev_col] = df[px_col] - 1.0
#         frames.append(df)
#     if not frames:
#         return pd.DataFrame(columns=["open_time"] + [c for pair in STABLE_MAP.values() for c in pair])
#     out = frames[0]
#     for f in frames[1:]:
#         out = out.merge(f, on="open_time", how="outer")
#     out = out.sort_values("open_time").reset_index(drop=True)
#     return out

# def compute_breadth(tf: str, spot_files: list[str]) -> pd.DataFrame:
#     """
#     Строит breadth из доступных спот-файлов *USDT (исключая стейблы):
#     breadth_up, breadth_down, breadth_flat, breadth_ad (up - down)
#     """
#     from collections import defaultdict
#     up = defaultdict(int); down = defaultdict(int); flat = defaultdict(int)

#     # отфильтруем только USDT (и не стейблы)
#     files = []
#     for p in spot_files:
#         # символ: .../spot_{SYMBOL}/{tf}/{SYMBOL}-{tf}.csv
#         sym = os.path.basename(os.path.dirname(os.path.dirname(p))).split("spot_")[-1]
#         if not sym.endswith("USDT"):
#             continue
#         if sym in STABLE_SYMBOLS:
#             continue
#         files.append(p)

#     if not files:
#         # пустой каркас
#         return pd.DataFrame(columns=["open_time","breadth_up","breadth_down","breadth_flat","breadth_ad"])

#     for path in files:
#         df = pd.read_csv(path, header=None, usecols=[0,4])  # open_time, close
#         df = df[pd.to_numeric(df[0], errors="coerce").notnull()]
#         df.columns = ["open_time","close"]
#         df = df.sort_values("open_time")
#         # pct-change по close
#         ch = df["close"].astype(float).pct_change()
#         for ts, val in zip(df["open_time"].values, ch.values):
#             if not np.isfinite(val):
#                 continue
#             if val > 0:   up[ts]   += 1
#             elif val < 0: down[ts] += 1
#             else:         flat[ts] += 1

#     # собрать в таблицу
#     all_ts = sorted(set(list(up.keys()) + list(down.keys()) + list(flat.keys())))
#     b_up   = [up.get(ts,0) for ts in all_ts]
#     b_down = [down.get(ts,0) for ts in all_ts]
#     b_flat = [flat.get(ts,0) for ts in all_ts]
#     b_ad   = [u - d for u, d in zip(b_up, b_down)]
#     out = pd.DataFrame({
#         "open_time": all_ts,
#         "breadth_up": b_up,
#         "breadth_down": b_down,
#         "breadth_flat": b_flat,
#         "breadth_ad": b_ad,
#     })
#     return out

# # ---------- Основной проход ----------
# def process_timeframe(tf: str):
#     spot_glob = f"{BASE_DIR}/spot_*/*/{'*' if tf == '' else tf}/*-{tf}.csv"
#     fut_glob  = f"{BASE_DIR}/fut_*/*/{'*' if tf == '' else tf}/*-{tf}.csv"
#     spot_files = glob.glob(f"{BASE_DIR}/spot_*{os.sep}{tf}{os.sep}*-{tf}.csv")
#     fut_files  = glob.glob(f"{BASE_DIR}/fut_*{os.sep}{tf}{os.sep}*-{tf}.csv")

#     # индексы путей по символам
#     def sym_from_path(p: str) -> str:
#         # /.../_crypto_data/spot_SYMBOL/{tf}/SYMBOL-{tf}.csv
#         base = os.path.basename(os.path.dirname(os.path.dirname(p)))
#         # base = 'spot_SYMBOL'
#         return base.split("spot_")[-1] if "spot_" in base else base.split("fut_")[-1]

#     spot_by_sym = {sym_from_path(p): p for p in spot_files}
#     fut_by_sym  = {sym_from_path(p): p for p in fut_files}
#     all_syms = sorted(set(spot_by_sym) | set(fut_by_sym))

#     # заготовки: стейблы/бредс
#     stable_pack = load_stable_pack(tf)
#     breadth_df  = compute_breadth(tf, spot_files)

#     for sym in all_syms:
#         spot_path = spot_by_sym.get(sym)
#         fut_path  = fut_by_sym.get(sym)

#         # заранее посчитаем flows по обеим сторонам (если файлы есть)
#         spot_flows = None
#         fut_flows  = None

#         if spot_path and os.path.exists(spot_path):
#             df_spot_raw = read_klines_csv(spot_path)
#             spot_flows = compute_flows_cols(df_spot_raw, prefix="spot")
#         if fut_path and os.path.exists(fut_path):
#             df_fut_raw = read_klines_csv(fut_path)
#             fut_flows  = compute_flows_cols(df_fut_raw,  prefix="perp")

#         # ---- пишем SPOT-файл (если есть) ----
#         if spot_path and os.path.exists(spot_path):
#             df = df_spot_raw.copy()
#             df = compute_price_ta(df)

#             # flows spot (свои)
#             if spot_flows is not None:
#                 df = df.merge(spot_flows, on="open_time", how="left")
#             else:
#                 for c in ["flows_spot_buy_q","flows_spot_sell_q","flows_spot_net_q"]:
#                     df[c] = np.nan

#             # flows perp (из фьюча, если есть)
#             if fut_flows is not None:
#                 df = df.merge(fut_flows, on="open_time", how="left")
#             else:
#                 for c in ["flows_perp_buy_q","flows_perp_sell_q","flows_perp_net_q"]:
#                     df[c] = np.nan

#             # stablecoins
#             if not stable_pack.empty:
#                 df = df.merge(stable_pack, on="open_time", how="left")
#             else:
#                 for _, (px_col, dev_col) in STABLE_MAP.items():
#                     df[px_col] = np.nan
#                     df[dev_col] = np.nan

#             # breadth
#             if not breadth_df.empty:
#                 df = df.merge(breadth_df, on="open_time", how="left")
#             else:
#                 for c in ["breadth_up","breadth_down","breadth_flat","breadth_ad"]:
#                     df[c] = np.nan

#             # порядок колонок
#             cols_order = KCOLS + [
#                 "ret_oc","ret_cc","rsi14","atr14",
#                 "flows_spot_buy_q","flows_spot_sell_q","flows_spot_net_q",
#                 "flows_perp_buy_q","flows_perp_sell_q","flows_perp_net_q",
#                 "usdc_px","usdc_dev","fdusd_px","fdusd_dev","usdp_px","usdp_dev",
#                 "breadth_up","breadth_down","breadth_flat","breadth_ad",
#             ]
#             # добавим отсутствующие (на случай пустых паков)
#             for c in cols_order:
#                 if c not in df.columns:
#                     df[c] = np.nan
#             df = df[cols_order]

#             write_csv_overwrite(df, spot_path)
#             print(f"[OK] SPOT saved: {spot_path}")

#         # ---- пишем FUT-файл (если есть) ----
#         if fut_path and os.path.exists(fut_path):
#             df = df_fut_raw.copy()
#             df = compute_price_ta(df)

#             # flows perp (свои)
#             if fut_flows is not None:
#                 df = df.merge(fut_flows, on="open_time", how="left")
#             else:
#                 for c in ["flows_perp_buy_q","flows_perp_sell_q","flows_perp_net_q"]:
#                     df[c] = np.nan

#             # flows spot (из спота, если есть)
#             if spot_flows is not None:
#                 # переименуем, чтобы не конфликтовали имена с уже добавленными перп
#                 df = df.merge(spot_flows, on="open_time", how="left")
#             else:
#                 for c in ["flows_spot_buy_q","flows_spot_sell_q","flows_spot_net_q"]:
#                     df[c] = np.nan

#             # stablecoins
#             if not stable_pack.empty:
#                 df = df.merge(stable_pack, on="open_time", how="left")
#             else:
#                 for _, (px_col, dev_col) in STABLE_MAP.items():
#                     df[px_col] = np.nan
#                     df[dev_col] = np.nan

#             # breadth
#             if not breadth_df.empty:
#                 df = df.merge(breadth_df, on="open_time", how="left")
#             else:
#                 for c in ["breadth_up","breadth_down","breadth_flat","breadth_ad"]:
#                     df[c] = np.nan

#             cols_order = KCOLS + [
#                 "ret_oc","ret_cc","rsi14","atr14",
#                 "flows_spot_buy_q","flows_spot_sell_q","flows_spot_net_q",
#                 "flows_perp_buy_q","flows_perp_sell_q","flows_perp_net_q",
#                 "usdc_px","usdc_dev","fdusd_px","fdusd_dev","usdp_px","usdp_dev",
#                 "breadth_up","breadth_down","breadth_flat","breadth_ad",
#             ]
#             for c in cols_order:
#                 if c not in df.columns:
#                     df[c] = np.nan
#             df = df[cols_order]

#             write_csv_overwrite(df, fut_path)
#             print(f"[OK] FUT saved:  {fut_path}")

# def main():
#     for tf in TIMEFRAMES:
#         print(f"=== timeframe: {tf} ===")
#         process_timeframe(tf)

# if __name__ == "__main__":
#     main()
