# -*- coding: utf-8 -*-
"""
intel.py — практичный сборщик рыночных сигналов без приватных ключей.

Исправлено:
- Потоки спот/перп теперь берутся через /aggTrades (startTime/endTime) без API-ключей.
- Учитываем флаг 'm' (isBuyerMaker) из aggTrades, а не 'isBuyerMaker' из /trades.

Крипто (Binance) + «внешняя погода»:
- ES (E-mini S&P 500), NQ (E-mini Nasdaq 100), DXY (ICE Dollar Index) через Stooq CSV-снимки.

Покрытие (публичные фиды):
  • Binance USDⓈ-M: цена/свечи, OI, funding, premium index (basis), aggTrades (спот/перп), книга заявок, futures data ratios.
  • Bybit/OKX/Deribit: кросс-биржевой снапшот перпов (mark/index/funding/basis).
  • Stooq (без ключей): ES.F, NQ.F, DX.F / USD_I — «сейчас» (OHLCV) и интрадей-изменение.

Зависимости: requests
"""

from __future__ import annotations

import time
import math
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests


# -------------------------
# Константы базовых URL
# -------------------------

BINANCE_SPOT = "https://api.binance.com"
BINANCE_FAPI = "https://fapi.binance.com"   # USDⓈ-M Futures
BYBIT = "https://api.bybit.com"             # v5 public
OKX = "https://www.okx.com"
DERIBIT = "https://www.deribit.com"
STOOQ = "https://stooq.com"                 # CSV quote snapshots (q/l)

# Таймауты/ретраи
REQ_TIMEOUT = 10
MAX_RETRIES = 2
SLEEP_BETWEEN_RETRIES = 0.25


# -------------------------
# Утилиты
# -------------------------

def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _pct(a: float, b: float) -> float:
    """(b - a)/a в долях; безопасно к NaN и нулю."""
    if not math.isfinite(a) or a == 0 or not math.isfinite(b):
        return float("nan")
    return (b - a) / a


class Http:
    """Простой HTTP-клиент с ретраями."""
    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")
        self.s = requests.Session()

    def get(self, path: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Any:
        url = f"{self.base}{path}"
        last_err = None
        for _ in range(MAX_RETRIES + 1):
            try:
                r = self.s.get(url, params=params, headers=headers or {}, timeout=REQ_TIMEOUT)
                r.raise_for_status()
                ctype = (r.headers.get("Content-Type") or "").lower()
                if "application/json" in ctype:
                    return r.json()
                return r.text
            except Exception as e:
                last_err = e
                time.sleep(SLEEP_BETWEEN_RETRIES)
        raise RuntimeError(f"GET failed {url} params={params}: {last_err}")


# -------------------------
# Клиенты бирж (публичные)
# -------------------------

class BinancePublic:
    """Публичные REST-эндпоинты Binance (spot + USDⓈ-M fapi)."""

    def __init__(self):
        self.spot = Http(BINANCE_SPOT)
        self.fapi = Http(BINANCE_FAPI)

    # ---- Свечи/цена/базис ----

    def fapi_klines(self, symbol: str, interval: str, start_time: int, end_time: int, limit: int = 500) -> List[list]:
        params = {"symbol": symbol, "interval": interval, "startTime": start_time, "endTime": end_time, "limit": limit}
        return self.fapi.get("/fapi/v1/klines", params)

    def fapi_premium_index(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        params = {}
        if symbol:
            params["symbol"] = symbol
        return self.fapi.get("/fapi/v1/premiumIndex", params)

    def fapi_premium_index_klines(self, symbol: str, interval: str, start_time: int, end_time: int, limit: int = 500) -> List[list]:
        params = {"symbol": symbol, "interval": interval, "startTime": start_time, "endTime": end_time, "limit": limit}
        return self.fapi.get("/fapi/v1/premiumIndexKlines", params)

    def fapi_continuous_klines(self, pair: str, contract_type: str, interval: str, start_time: int, end_time: int, limit: int = 500) -> List[list]:
        params = {"pair": pair, "contractType": contract_type, "interval": interval, "startTime": start_time, "endTime": end_time, "limit": limit}
        return self.fapi.get("/fapi/v1/continuousKlines", params)

    def fapi_index_price_klines(self, pair: str, interval: str, start_time: int, end_time: int, limit: int = 500) -> List[list]:
        params = {"pair": pair, "interval": interval, "startTime": start_time, "endTime": end_time, "limit": limit}
        return self.fapi.get("/fapi/v1/indexPriceKlines", params)

    # ---- OI / ratios ----

    def fapi_open_interest_hist(self, symbol: str, period: str, start_time: int, end_time: int, limit: int = 200) -> List[Dict[str, Any]]:
        params = {"symbol": symbol, "period": period, "limit": limit, "startTime": start_time, "endTime": end_time}
        return self.fapi.get("/futures/data/openInterestHist", params)

    def futures_data_taker_long_short_ratio(self, symbol: str, period: str, limit: int = 30, start_time: Optional[int] = None, end_time: Optional[int] = None) -> List[Dict[str, Any]]:
        params = {"symbol": symbol, "period": period, "limit": limit}
        if start_time: params["startTime"] = start_time
        if end_time: params["endTime"] = end_time
        return self.fapi.get("/futures/data/takerlongshortRatio", params)

    def futures_data_global_long_short_account_ratio(self, symbol: str, period: str, limit: int = 30, start_time: Optional[int] = None, end_time: Optional[int] = None) -> List[Dict[str, Any]]:
        params = {"symbol": symbol, "period": period, "limit": limit}
        if start_time: params["startTime"] = start_time
        if end_time: params["endTime"] = end_time
        return self.fapi.get("/futures/data/globalLongShortAccountRatio", params)

    def futures_data_top_long_short_account_ratio(self, symbol: str, period: str, limit: int = 30, start_time: Optional[int] = None, end_time: Optional[int] = None) -> List[Dict[str, Any]]:
        params = {"symbol": symbol, "period": period, "limit": limit}
        if start_time: params["startTime"] = start_time
        if end_time: params["endTime"] = end_time
        return self.fapi.get("/futures/data/topLongShortAccountRatio", params)

    def futures_data_top_long_short_position_ratio(self, symbol: str, period: str, limit: int = 30, start_time: Optional[int] = None, end_time: Optional[int] = None) -> List[Dict[str, Any]]:
        params = {"symbol": symbol, "period": period, "limit": limit}
        if start_time: params["startTime"] = start_time
        if end_time: params["endTime"] = end_time
        return self.fapi.get("/futures/data/topLongShortPositionRatio", params)

    # ---- Лента/книга/тикеры ----

    def spot_agg_trades(self, symbol: str, start_time: int, end_time: int, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        GET /api/v3/aggTrades — публично, без ключа.
        Параметры: symbol, startTime, endTime, limit<=1000
        """
        params = {"symbol": symbol, "startTime": start_time, "endTime": end_time, "limit": min(limit, 1000)}
        return self.spot.get("/api/v3/aggTrades", params)

    def fapi_agg_trades(self, symbol: str, start_time: int, end_time: int, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        GET /fapi/v1/aggTrades — публично, без ключа.
        Параметры: symbol, startTime, endTime, limit<=1000
        """
        params = {"symbol": symbol, "startTime": start_time, "endTime": end_time, "limit": min(limit, 1000)}
        return self.fapi.get("/fapi/v1/aggTrades", params)

    def spot_depth(self, symbol: str, limit: int = 5000) -> Dict[str, Any]:
        params = {"symbol": symbol, "limit": min(limit, 5000)}
        return self.spot.get("/api/v3/depth", params)

    def spot_ticker_24hr_all(self) -> List[Dict[str, Any]]:
        return self.spot.get("/api/v3/ticker/24hr", None)

    def spot_ticker_price(self, symbol: str) -> Dict[str, Any]:
        return self.spot.get("/api/v3/ticker/price", {"symbol": symbol})


class BybitPublic:
    def __init__(self):
        self.http = Http(BYBIT)
    def linear_ticker(self, symbol: str) -> Dict[str, Any]:
        data = self.http.get("/v5/market/tickers", {"category": "linear", "symbol": symbol})
        if isinstance(data, dict) and "result" in data and "list" in data["result"] and data["result"]["list"]:
            return data["result"]["list"][0]
        return {}


class OkxPublic:
    def __init__(self):
        self.http = Http(OKX)
    def swap_ticker(self, inst_id: str = "BTC-USDT-SWAP") -> Dict[str, Any]:
        data = self.http.get("/api/v5/market/ticker", {"instId": inst_id})
        if isinstance(data, dict) and "data" in data and data["data"]:
            return data["data"][0]
        return {}
    def funding_rate(self, inst_id: str = "BTC-USDT-SWAP") -> Dict[str, Any]:
        data = self.http.get("/api/v5/public/funding-rate", {"instId": inst_id})
        if isinstance(data, dict) and "data" in data and data["data"]:
            return data["data"][0]
        return {}


class DeribitPublic:
    def __init__(self):
        self.http = Http(DERIBIT)
    def book_summary_perpetual(self, instrument: str = "BTC-PERPETUAL") -> Dict[str, Any]:
        data = self.http.get("/api/v2/public/get_book_summary_by_instrument", {"instrument_name": instrument})
        if isinstance(data, dict) and "result" in data and data["result"]:
            return data["result"][0]
        return {}


class StooqPublic:
    """
    Stooq CSV-«снимок» котировок: /q/l/?s=SYMBOL&f=sd2t2ohlcv&h&e=csv
    Поля: symbol, date(yyyy-mm-dd), time(hh:mm:ss), open, high, low, close, volume.
    """
    def __init__(self):
        self.http = Http(STOOQ)

    def quote_snapshot(self, symbol: str) -> Dict[str, Any]:
        path = "/q/l/"
        params = {"s": symbol.lower(), "f": "sd2t2ohlcv", "h": "", "e": "csv"}
        txt = self.http.get(path, params)
        if not isinstance(txt, str) or not txt.strip():
            return {}
        lines = [ln.strip() for ln in txt.strip().splitlines() if ln.strip()]
        if len(lines) < 2:
            return {}
        header = [h.strip().lower() for h in lines[0].split(",")]
        row = [c.strip() for c in lines[1].split(",")]
        idx = {name: i for i, name in enumerate(header)}
        def col(name: str) -> Optional[str]:
            i = idx.get(name)
            return row[i] if i is not None and i < len(row) else None
        o = _safe_float(col("open")); h = _safe_float(col("high"))
        l = _safe_float(col("low"));  c = _safe_float(col("close"))
        v = _safe_float(col("volume"))
        date_s = col("date") or ""
        time_s = col("time") or ""
        return {
            "symbol": col("symbol") or symbol.upper(),
            "date": date_s,
            "time": time_s,
            "open": o, "high": h, "low": l, "close": c, "volume": v,
            "intraday_change_pct": _pct(o, c),
            "valid": (math.isfinite(o) and math.isfinite(c)),
        }


# -------------------------
# Вспомогательные вычисления
# -------------------------

def _sum_quote_from_aggtrades(trades: List[Dict[str, Any]]) -> Tuple[float, float]:
    """
    Суммирует объём в котируемой валюте по aggTrades (spot/futures).
    'm' == True  -> buyer is maker  -> сделка продавца-такера (taker SELL)
    'm' == False -> buyer is taker  -> сделка покупателя-такера (taker BUY)
    """
    buy_q, sell_q = 0.0, 0.0
    for t in trades:
        price = _safe_float(t.get("p") or t.get("price"))
        qty   = _safe_float(t.get("q") or t.get("qty") or t.get("baseQty"))
        if not (math.isfinite(price) and math.isfinite(qty)):
            continue
        quote = price * qty
        is_buyer_maker = bool(t.get("m"))  # ключ из aggTrades
        if is_buyer_maker:
            sell_q += quote   # taker SELL
        else:
            buy_q += quote    # taker BUY
    return buy_q, sell_q


def _orderbook_tilt(depth: Dict[str, Any], mid: float, pct_radius: float) -> Dict[str, float]:
    """Суммирует заявки в кольце +- pct_radius от mid."""
    hi = mid * (1 + pct_radius)
    lo = mid * (1 - pct_radius)
    bids = depth.get("bids") or []
    asks = depth.get("asks") or []
    bid_vol = 0.0
    ask_vol = 0.0
    for p, q in bids:
        fp, fq = _safe_float(p), _safe_float(q)
        if math.isfinite(fp) and math.isfinite(fq) and lo <= fp <= hi:
            bid_vol += fq
    for p, q in asks:
        fp, fq = _safe_float(p), _safe_float(q)
        if math.isfinite(fp) and math.isfinite(fq) and lo <= fp <= hi:
            ask_vol += fq
    tilt = (bid_vol / ask_vol) if (ask_vol > 0) else float("inf")
    return {"bid_vol_in_band": bid_vol, "ask_vol_in_band": ask_vol, "tilt_bid_over_ask": tilt}


def _interval_to_ms(period: str) -> int:
    period = period.strip().lower()
    if period.endswith("ms"): return int(period[:-2])
    if period.endswith("s"):  return int(period[:-1]) * 1000
    if period.endswith("m"):  return int(period[:-1]) * 60_000
    if period.endswith("h"):  return int(period[:-1]) * 3_600_000
    if period.endswith("d"):  return int(period[:-1]) * 86_400_000
    raise ValueError(f"Unknown period: {period}")


# -------------------------
# Основной фасад
# -------------------------

class MarketIntel:
    """Фасад для разового снимка сигналов."""

    def __init__(self):
        self.binance = BinancePublic()
        self.bybit = BybitPublic()
        self.okx = OkxPublic()
        self.deribit = DeribitPublic()
        self.stooq = StooqPublic()

    # ---- БАЗОВЫЕ БЛОКИ Binance ----

    def price_block(self, symbol: str, lookback_hours: float, interval: str = "5m") -> Dict[str, Any]:
        end = _now_ms()
        start = end - int(lookback_hours * 3600_000)
        kl = self.binance.fapi_klines(symbol, interval, start, end)
        if not kl:
            return {}
        o = _safe_float(kl[0][1])
        c = _safe_float(kl[-1][4])
        chg = _pct(o, c)
        return {"open": o, "close": c, "change_pct": chg, "bars": len(kl), "t_start": kl[0][0], "t_end": kl[-1][6] if len(kl[-1]) > 6 else kl[-1][0]}

    def open_interest_block(self, symbol: str, lookback_hours: float, period: str = "5m") -> Dict[str, Any]:
        end = _now_ms()
        start = end - int(lookback_hours * 3600_000)
        hist = self.binance.fapi_open_interest_hist(symbol, period, start, end, limit=200)
        if not hist:
            return {}
        oi_then = _safe_float(hist[0].get("sumOpenInterest"))
        oi_now = _safe_float(hist[-1].get("sumOpenInterest"))
        chg = _pct(oi_then, oi_now)
        return {"oi_then": oi_then, "oi_now": oi_now, "oi_change_pct": chg, "points": len(hist), "t_start": int(hist[0].get("timestamp") or 0), "t_end": int(hist[-1].get("timestamp") or 0)}

    def funding_basis_block(self, symbol: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        pi = self.binance.fapi_premium_index(symbol)
        end = _now_ms()
        start = end - 3600_000
        pik = self.binance.fapi_premium_index_klines(symbol, "5m", start, end, limit=24)
        basis_last_close = _safe_float(pik[-1][4]) if pik else float("nan")
        basis_then_open = _safe_float(pik[0][1]) if pik else float("nan")
        basis_now = float("nan")
        if pi:
            mark = _safe_float(pi.get("markPrice"))
            index = _safe_float(pi.get("indexPrice"))
            if math.isfinite(mark) and math.isfinite(index) and index != 0:
                basis_now = (mark - index) / index
        funding_rate = _safe_float(pi.get("lastFundingRate")) if pi else float("nan")
        who_pays = None
        if math.isfinite(funding_rate):
            who_pays = "longs_pay_shorts" if funding_rate > 0 else "shorts_pay_longs" if funding_rate < 0 else "neutral"
        funding = {
            "rates": [],
            "avg_rate": float("nan"),
            "last_funding_rate": funding_rate,
            "who_pays_now": who_pays,
            "mark_price": _safe_float(pi.get("markPrice")) if pi else float("nan"),
            "index_price": _safe_float(pi.get("indexPrice")) if pi else float("nan"),
            "snapshot_time": int(pi.get("time") or 0) if pi else None,
        }
        basis = {
            "basis_now": basis_now,
            "basis_last_close": basis_last_close,
            "basis_then_open": basis_then_open,
            "basis_change_abs": (basis_now - basis_then_open) if (math.isfinite(basis_now) and math.isfinite(basis_then_open)) else float("nan"),
            "bars": len(pik),
        }
        return funding, basis

    def _collect_agg_trades_timeboxed(self, fetch_fn, symbol: str, lookback_hours: float, step_minutes: int = 15) -> List[Dict[str, Any]]:
        """
        Собираем aggTrades публично, без ключей, нарезая окно на интервалы времени.
        Если в интервале возвращается ровно 1000 записей (лимит), динамически уменьшаем шаг.
        """
        end_ms = _now_ms()
        start_ms = end_ms - int(lookback_hours * 3600_000)

        results: List[Dict[str, Any]] = []
        step_ms = step_minutes * 60_000

        cur_start = start_ms
        while cur_start < end_ms:
            cur_end = min(end_ms, cur_start + step_ms)
            page = fetch_fn(symbol, start_time=cur_start, end_time=cur_end, limit=1000) or []
            results.extend(page)

            # Если выбили ровно 1000 — вероятно, отсечка по лимиту; уменьшим шаг в 3 раза и повторим участок
            if len(page) >= 1000 and step_ms > 60_000:
                step_ms = max(step_ms // 3, 60_000)  # минимум 1 минута
                continue

            # Иначе двигаем окно
            cur_start = cur_end

        return [t for t in results if int(t.get("T") or t.get("time") or 0) >= start_ms]

    def flows_block(self, spot_symbol: str, perp_symbol: str, lookback_hours: float) -> Dict[str, Any]:
        # СПОТ и ПЕРП: только aggTrades (startTime/endTime), без fromId и без ключей
        spot_tr = self._collect_agg_trades_timeboxed(self.binance.spot_agg_trades, spot_symbol, lookback_hours, step_minutes=15)
        perp_tr = self._collect_agg_trades_timeboxed(self.binance.fapi_agg_trades, perp_symbol, lookback_hours, step_minutes=15)

        sb, ss = _sum_quote_from_aggtrades(spot_tr)
        pb, ps = _sum_quote_from_aggtrades(perp_tr)

        spot_net = sb - ss
        perp_net = pb - ps
        return {
            "spot": {
                "taker_buy_quote": sb, "taker_sell_quote": ss, "taker_net_quote": spot_net,
                "sense": "net_taker_buy" if spot_net > 0 else "net_taker_sell" if spot_net < 0 else "balanced",
            },
            "perp": {
                "taker_buy_quote": pb, "taker_sell_quote": ps, "taker_net_quote": perp_net,
                "sense": "net_taker_buy" if perp_net > 0 else "net_taker_sell" if perp_net < 0 else "balanced",
            },
            "spot_vs_perp": {
                "spot_net_minus_perp_net": spot_net - perp_net,
                "spot_stronger_than_perp": (spot_net > perp_net),
            }
        }

    def orderbook_block(self, spot_symbol: str, use_price: Optional[float] = None) -> Dict[str, Any]:
        depth = self.binance.spot_depth(spot_symbol, limit=5000)
        best_bid = _safe_float(depth["bids"][0][0]) if depth.get("bids") else float("nan")
        best_ask = _safe_float(depth["asks"][0][0]) if depth.get("asks") else float("nan")
        mid = use_price if (use_price and math.isfinite(use_price)) else (best_bid + best_ask) / 2.0
        bands = {}
        for pct_band in (0.005, 0.01):  # 0.5% и 1%
            k = f"{pct_band*100:.2f}%"
            bands[k] = _orderbook_tilt(depth, mid, pct_band)
        return {"mid": mid, "bands": bands}

    # ---- НОВЫЕ БЛОКИ (кросс-биржи/календарный базис/рацио/ширина/стейблы) ----

    def cross_exchange_perp_snapshot(self, symbol_binance: str = "BTCUSDT",
                                     bybit_symbol: str = "BTCUSDT",
                                     okx_inst: str = "BTC-USDT-SWAP",
                                     deribit_instr: str = "BTC-PERPETUAL") -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        # Binance
        b = self.binance.fapi_premium_index(symbol_binance)
        if b:
            mark = _safe_float(b.get("markPrice")); index = _safe_float(b.get("indexPrice"))
            basis = (mark - index) / index if (math.isfinite(mark) and math.isfinite(index) and index) else float("nan")
            fr = _safe_float(b.get("lastFundingRate"))
            out["binance"] = {"mark": mark, "index": index, "basis": basis,
                              "fundingRate": fr,
                              "whoPays": "longs_pay_shorts" if (math.isfinite(fr) and fr > 0) else "shorts_pay_longs" if (math.isfinite(fr) and fr < 0) else "neutral",
                              "ts": int(b.get("time") or 0)}

        # Bybit
        y = self.bybit.linear_ticker(bybit_symbol)
        if y:
            mark = _safe_float(y.get("markPrice")); index = _safe_float(y.get("indexPrice"))
            basis = (mark - index) / index if (math.isfinite(mark) and math.isfinite(index) and index) else float("nan")
            fr = _safe_float(y.get("fundingRate"))
            out["bybit"] = {"mark": mark, "index": index, "basis": basis,
                            "fundingRate": fr,
                            "whoPays": "longs_pay_shorts" if (math.isfinite(fr) and fr > 0) else "shorts_pay_longs" if (math.isfinite(fr) and fr < 0) else "neutral",
                            "ts": int(y.get("ts") or 0)}

        # OKX
        o_ticker = self.okx.swap_ticker(okx_inst)
        o_funding = self.okx.funding_rate(okx_inst)
        if o_ticker:
            mark = _safe_float(o_ticker.get("markPx")); index = _safe_float(o_ticker.get("idxPx"))
            basis = (mark - index) / index if (math.isfinite(mark) and math.isfinite(index) and index) else float("nan")
            fr = _safe_float((o_funding or {}).get("fundingRate"))
            out["okx"] = {"mark": mark, "index": index, "basis": basis,
                          "fundingRate": fr,
                          "whoPays": "longs_pay_shorts" if (math.isfinite(fr) and fr > 0) else "shorts_pay_longs" if (math.isfinite(fr) and fr < 0) else "neutral",
                          "ts": int(o_ticker.get("ts") or 0)}

        # Deribit
        d = self.deribit.book_summary_perpetual(deribit_instr)
        if d:
            mark = _safe_float(d.get("mark_price")); index = _safe_float(d.get("index_price"))
            basis = (mark - index) / index if (math.isfinite(mark) and math.isfinite(index) and index) else float("nan")
            out["deribit"] = {"mark": mark, "index": index, "basis": basis, "fundingRate": None, "whoPays": None, "ts": int(d.get("timestamp") or 0)}
        return out

    def calendar_basis_block(self, pair: str = "BTCUSDT", interval: str = "5m", lookback_hours: float = 2.0) -> Dict[str, Any]:
        end = _now_ms()
        start = end - int(lookback_hours * 3600_000)

        def last_basis(contract_type: str) -> Dict[str, Any]:
            fut = self.binance.fapi_continuous_klines(pair, contract_type, interval, start, end, limit=200)
            idx = self.binance.fapi_index_price_klines(pair, interval, start, end, limit=200)
            if not fut or not idx:
                return {"basis_now": float("nan"), "basis_then_open": float("nan"), "bars": 0}
            f_close = _safe_float(fut[-1][4]); i_close = _safe_float(idx[-1][4])
            f_open0 = _safe_float(fut[0][1]);  i_open0 = _safe_float(idx[0][1])
            now = (f_close - i_close) / i_close if (math.isfinite(f_close) and math.isfinite(i_close) and i_close) else float("nan")
            then = (f_open0 - i_open0) / i_open0 if (math.isfinite(f_open0) and math.isfinite(i_open0) and i_open0) else float("nan")
            return {"basis_now": now, "basis_then_open": then, "basis_change_abs": (now - then) if math.isfinite(now) and math.isfinite(then) else float("nan"), "bars": min(len(fut), len(idx))}
        return {"current_quarter": last_basis("CURRENT_QUARTER"), "next_quarter": last_basis("NEXT_QUARTER")}

    def sentiment_ratios_block(self, symbol: str, period: str = "5m", lookback_points: int = 24) -> Dict[str, Any]:
        end = _now_ms()
        start = end - (lookback_points * _interval_to_ms(period))
        taker = self.binance.futures_data_taker_long_short_ratio(symbol, period, limit=lookback_points, start_time=start, end_time=end)
        glob  = self.binance.futures_data_global_long_short_account_ratio(symbol, period, limit=lookback_points, start_time=start, end_time=end)
        top_a = self.binance.futures_data_top_long_short_account_ratio(symbol, period, limit=lookback_points, start_time=start, end_time=end)
        top_p = self.binance.futures_data_top_long_short_position_ratio(symbol, period, limit=lookback_points, start_time=start, end_time=end)

        def last_ratio(arr: List[Dict[str, Any]], field: str) -> Optional[float]:
            if not arr: return None
            return _safe_float(arr[-1].get(field) or arr[-1].get("buySellRatio"))
        return {
            "taker_buy_sell_ratio": last_ratio(taker, "buySellRatio"),
            "global_long_short_ratio": last_ratio(glob, "longShortRatio"),
            "top_trader_accounts_ratio": last_ratio(top_a, "longShortRatio"),
            "top_trader_positions_ratio": last_ratio(top_p, "longShortRatio"),
            "points": {"taker": len(taker), "global": len(glob), "top_accounts": len(top_a), "top_positions": len(top_p)}
        }

    def market_breadth_spot_usdt(self, top_n_by_quote_vol: int = 50) -> Dict[str, Any]:
        arr = self.binance.spot_ticker_24hr_all()
        usdt = [r for r in arr if isinstance(r, dict) and str(r.get("symbol", "")).endswith("USDT")]
        usdt.sort(key=lambda x: _safe_float(x.get("quoteVolume"), 0.0), reverse=True)
        top = usdt[:top_n_by_quote_vol]
        up = sum(1 for r in top if _safe_float(r.get("priceChangePercent")) > 0)
        down = sum(1 for r in top if _safe_float(r.get("priceChangePercent")) < 0)
        flat = len(top) - up - down
        return {"universe": len(usdt), "considered": len(top), "up": up, "down": down, "flat": flat, "advance_decline": up - down}

    def stablecoin_deviation(self, symbols: List[str] = ("USDCUSDT", "FDUSDUSDT", "USDPUSDT")) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for s in symbols:
            try:
                px = self.binance.spot_ticker_price(s); p = _safe_float(px.get("price"))
                out[s] = {"last": p, "deviation_from_1": (p - 1.0) if math.isfinite(p) else float("nan")}
            except Exception:
                out[s] = {"last": float("nan"), "deviation_from_1": float("nan")}
        return out

    # ---- ВНЕШНЯЯ ПОГОДА (ES/NQ/DXY через Stooq) ----

    def macro_weather_block(self) -> Dict[str, Any]:
        """
        Снимок ES, NQ, DXY:
          • ES.F — E-mini S&P 500 (CME)
          • NQ.F — E-mini Nasdaq-100 (CME)
          • DX.F — ICE US Dollar Index futures (фолбэк: USD_I)
        Данные: Stooq CSV snapshot (symbol,date,time,open,high,low,close,volume).
        """
        def get_one(sym: str) -> Dict[str, Any]:
            try:
                q = self.stooq.quote_snapshot(sym)
                if not q or not q.get("valid"):
                    return {"symbol": sym.upper(), "ok": False}
                intr = q.get("intraday_change_pct")
                sense = "up" if (isinstance(intr, float) and intr > 0) else "down" if (isinstance(intr, float) and intr < 0) else "flat"
                return {
                    "symbol": q["symbol"], "date": q["date"], "time": q["time"],
                    "open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"], "volume": q["volume"],
                    "intraday_change_pct": intr, "sense": sense, "ok": True
                }
            except Exception:
                return {"symbol": sym.upper(), "ok": False}

        es = get_one("ES.F")
        nq = get_one("NQ.F")
        dx = get_one("DX.F")
        if not dx.get("ok"):
            dx = get_one("USD_I")  # фолбэк

        def lean(sense: Optional[str]) -> int:
            return 1 if sense == "up" else -1 if sense == "down" else 0

        regressor = lean(es.get("sense")) + lean(nq.get("sense")) - lean(dx.get("sense"))
        return {"ES": es, "NQ": nq, "DXY": dx, "macro_lean_score": regressor}

    # ---- Компоновка полного снапшота ----

    def snapshot(self, symbol: str = "BTCUSDT", lookback_hours: float = 2.0,
                 asof_utc: Optional[datetime] = None) -> Dict[str, Any]:
        if asof_utc is None:
            asof_utc = datetime.now(timezone.utc)

        # Базовые блоки (Binance)
        price = self.price_block(symbol, lookback_hours, interval="5m")
        open_interest = self.open_interest_block(symbol, lookback_hours, period="5m")
        funding, basis = self.funding_basis_block(symbol)
        flows = self.flows_block(spot_symbol="BTCUSDT", perp_symbol=symbol, lookback_hours=lookback_hours)
        orderbook = self.orderbook_block(spot_symbol="BTCUSDT", use_price=price.get("close") if price else None)

        # Новые блоки
        x_perp = self.cross_exchange_perp_snapshot(symbol_binance=symbol, bybit_symbol="BTCUSDT", okx_inst="BTC-USDT-SWAP", deribit_instr="BTC-PERPETUAL")
        cal_basis = self.calendar_basis_block(pair="BTCUSDT", interval="5m", lookback_hours=lookback_hours)
        sent = self.sentiment_ratios_block(symbol=symbol, period="5m", lookback_points=int(lookback_hours * 12))
        breadth = self.market_breadth_spot_usdt(top_n_by_quote_vol=50)
        stables = self.stablecoin_deviation()
        macro = self.macro_weather_block()  # ES/NQ/DXY

        # Подсказки (минимальная эвристика)
        hints: List[str] = []
        if open_interest and isinstance(open_interest.get("oi_change_pct"), float):
            if open_interest["oi_change_pct"] < 0:
                hints.append("Снижение OI — движение может быть закрытием позиций (хуже для продолжения).")
            elif open_interest["oi_change_pct"] > 0:
                hints.append("Рост OI — чаще набор новых позиций (устойчивее).")
        if basis and isinstance(basis.get("basis_change_abs"), float) and math.isfinite(basis["basis_change_abs"]):
            if basis["basis_change_abs"] > 0:
                hints.append("Базис расширяется вверх — поддерживает бычий сценарий.")
            elif basis["basis_change_abs"] < 0:
                hints.append("Базис сжимается — осторожность.")
        if funding and isinstance(funding.get("last_funding_rate"), float) and math.isfinite(funding["last_funding_rate"]):
            if funding["last_funding_rate"] > 0 and flows.get("perp", {}).get("taker_net_quote", 0) > 0:
                hints.append("Рост на перпах при положительном funding — движение может быть хрупким.")
        if isinstance(macro.get("macro_lean_score"), int):
            if macro["macro_lean_score"] > 0:
                hints.append("ES/NQ вверх и/или DXY вниз — внешний фон поддерживает рост крипто.")
            elif macro["macro_lean_score"] < 0:
                hints.append("ES/NQ вниз и/или DXY вверх — внешний фон против роста крипто.")

        return {
            "asof_utc": asof_utc.isoformat(),
            "symbol": symbol,
            "lookback_hours": lookback_hours,
            "price": price,
            "open_interest": open_interest,
            "funding": funding,
            "basis": basis,
            "flows": flows,
            "orderbook": orderbook,
            # Новые секции:
            "x_perp": x_perp,
            "calendar_basis": cal_basis,
            "sentiment": sent,
            "breadth": breadth,
            "stablecoins": stables,
            "macro_weather": macro,
            "hints": hints,
        }

