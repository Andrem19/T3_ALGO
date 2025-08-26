
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, Iterator, List, Tuple, Any, Optional
import requests
import os
import time
import copy
import math
import argparse
import json
import shared_vars as sv
import vizualizer.viz as viz
import statistic.stat as stat
import vizualizer.correlation as cor

import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import requests
import train as tr


BINANCE_BASE = "https://api.binance.com"
KLINES_PATH = "/api/v3/klines"


def _parse_time_utc_to_str(ts: str) -> Optional[Tuple[datetime, str]]:
    """
    Парсит время из строки (например, '2025-08-22 11:00:00+00:00' или '...Z')
    и возвращает кортеж: (datetime_utc, 'YY-MM-DD HH:MM:SS').
    При отсутствии или ошибке парсинга — None.
    """
    if not ts:
        return None
    s = ts.strip()
    # Нормализуем суффикс 'Z' -> '+00:00' для совместимости с fromisoformat
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt: Optional[datetime] = None

    # 1) ISO с таймзоной
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        dt = None

    # 2) Без таймзоны 'YYYY-MM-DD HH:MM:SS'
    if dt is None:
        try:
            dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
            dt = dt.replace(tzinfo=timezone.utc)
        except Exception:
            return None

    # Гарантируем UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt_utc = dt.astimezone(timezone.utc)

    # Формат 'YY-MM-DD HH:MM:SS' (пример: '23-08-25 00:00:00')
    pretty = dt_utc.strftime("%y-%m-%d %H:%M:%S")
    return dt_utc, pretty

def _extract_score(value: Any) -> Optional[float]:
    """
    Унифицированное извлечение численного 'score':
    - Если value — словарь, возвращает value.get('score').
    - Если value — число (int/float), возвращает как есть.
    - Иначе None.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get("score")
    if isinstance(value, (int, float)):
        return float(value)
    return None

def load_compact_metrics(path: str) -> List[Dict[str, Any]]:
    """
    Читает JSON-Lines файл `path` (каждая строка — отдельный JSON-словарь),
    отбирает записи, где в per_metric есть news_score, rr25 и iv,
    и возвращает список компактных словарей с ключами:

        time, price_oi_funding, basis, flows, orderbook, cross,
        calendar, sentiment, breadth, stables, macro,
        news_score, rr25, iv, overall

    Список отсортирован по времени (от начала к концу).
    Значения метрик — это поля 'score' соответствующих подпоказателей,
    если они заданы словарями; если какие-то из необязательных метрик отсутствуют,
    в итоговом словаре для них будет None.
    """
    required_keys = ("news_score", "rr25", "iv")
    result: List[Tuple[datetime, Dict[str, Any]]] = []

    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                # Некорректная строка — пропускаем
                continue

            per = obj.get("per_metric") or {}

            # Проверяем наличие всех трёх обязательных полей
            if not all(k in per for k in required_keys):
                continue

            news_score = _extract_score(per.get("news_score"))
            rr25 = _extract_score(per.get("rr25"))
            iv = _extract_score(per.get("iv"))

            if news_score is None or rr25 is None or iv is None:
                # Если любое из обязательных значений не извлеклось — пропуск
                continue

            # Время
            ts_raw = obj.get("time_utc") or obj.get("time") or obj.get("timestamp")
            parsed = _parse_time_utc_to_str(ts_raw)
            if parsed is None:
                continue
            dt_utc, time_str = parsed

            # Удобный accessor
            def m(key: str) -> Optional[float]:
                return _extract_score(per.get(key))

            compact = {
                "time": time_str,
                "price_oi_funding": m("price_oi_funding"),
                "basis": m("basis"),
                "flows": m("flows"),
                "orderbook": m("orderbook"),
                "cross": m("cross"),
                "calendar": m("calendar"),
                "sentiment": m("sentiment"),
                "breadth": m("breadth"),
                "stables": m("stables"),
                "macro": m("macro"),
                "news_score": news_score,
                "rr25": rr25,
                "iv": iv,
                "overall": _extract_score((obj.get("overall") or {})),
            }

            result.append((dt_utc, compact))

    # Сортировка по времени: от ранних к поздним
    result.sort(key=lambda x: x[0])
    return [item for _, item in result]



TIME_FORMAT_INPUT = "%y-%m-%d %H:%M:%S"  # пример: '25-08-22 11:00:00' -> 2025-08-22 11:00:00 UTC

def parse_utc_time(s: str) -> datetime:
    """
    Парсит строку времени в UTC (timezone-aware).
    По умолчанию ожидает формат '%y-%m-%d %H:%M:%S'.
    """
    dt_naive = datetime.strptime(s, TIME_FORMAT_INPUT)
    return dt_naive.replace(tzinfo=timezone.utc)

def to_ms(dt: datetime) -> int:
    """Datetime (UTC-aware) -> миллисекунды UNIX."""
    if dt.tzinfo is None:
        # страховка: считаем вход UTC
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return int(dt.timestamp() * 1000)

def from_ms(ms: int) -> datetime:
    """Миллисекунды UNIX -> datetime UTC."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)

# ---------------------------------------------------------------------------
# Нормализация klines из Binance (работает и для list, и для dict)
# ---------------------------------------------------------------------------

def kline_open_time_ms(k: Any) -> int:
    """
    Возвращает open_time в миллисекундах для элемента свечи.
    Поддерживает оба формата:
      - list/tuple как у Binance REST: [openTime, open, high, low, close, volume, closeTime, ...]
      - dict с ключами 'openTime'/'open_time'
    """
    if isinstance(k, (list, tuple)):
        return int(k[0])
    if isinstance(k, dict):
        if "openTime" in k:
            return int(k["openTime"])
        if "open_time" in k:
            return int(k["open_time"])
    raise ValueError("Не удалось определить open_time у элемента kline")

def kline_to_dict(k: Any) -> Dict[str, Any]:
    """
    Приводит свечу к унифицированному словарю.
    Для list/tuple берём стандартный порядок Binance.
    """
    if isinstance(k, (list, tuple)):
        return {
            "open_time": int(k[0]),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
            "close_time": int(k[6]),
            # Остальные поля (quote asset volume, trades, taker buy volume, etc.) при наличии:
            "qav": float(k[7]) if len(k) > 7 else None,
            "num_trades": int(k[8]) if len(k) > 8 else None,
            "taker_base_vol": float(k[9]) if len(k) > 9 else None,
            "taker_quote_vol": float(k[10]) if len(k) > 10 else None,
        }
    if isinstance(k, dict):
        # Нормализуем ключи, где возможно
        open_time = int(k.get("openTime", k.get("open_time")))
        close_time = int(k.get("closeTime", k.get("close_time", open_time + 3600_000 - 1)))
        return {
            "open_time": open_time,
            "open": float(k.get("open")),
            "high": float(k.get("high")),
            "low": float(k.get("low")),
            "close": float(k.get("close")),
            "volume": float(k.get("volume", 0.0)),
            "close_time": close_time,
            "qav": float(k.get("qav", 0.0)),
            "num_trades": int(k.get("num_trades", 0)),
            "taker_base_vol": float(k.get("taker_base_vol", 0.0)),
            "taker_quote_vol": float(k.get("taker_quote_vol", 0.0)),
        }
    raise ValueError("Неизвестный формат свечи")

# ---------------------------------------------------------------------------
# Основная логика
# ---------------------------------------------------------------------------

def compute_time_bounds(metrics: List[Dict[str, Any]]) -> Tuple[datetime, datetime, bool]:
    """
    Возвращает (min_time, max_time, is_monotonic_non_decreasing)
    Все времена UTC-aware.
    """
    if not metrics:
        raise ValueError("Список метрик пуст")

    times = []
    for i, m in enumerate(metrics):
        t = parse_utc_time(m["time"])
        times.append(t)

    min_t = min(times)
    max_t = max(times)

    # Проверка монотонности исходного списка
    is_mono = all(times[i] <= times[i + 1] for i in range(len(times) - 1))
    return min_t, max_t, is_mono


def fetch_binance_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    session,
    limit_per_req: int = 1000,
    pause_sec: float = 0.5,
    max_retries: int = 5,
):
    """
    Качает klines c Binance SPOT (/api/v3/klines) за интервал [start_ms, end_ms] (оба включительно по open_time).
    Возвращает список свечей в «сыром» формате Binance: список списков:
        [openTime, open, high, low, close, volume, closeTime, quoteAssetVolume,
         numberOfTrades, takerBuyBaseAssetVolume, takerBuyQuoteAssetVolume, ignore]

    Аргументы:
        symbol         : например, "BTCUSDT"
        interval       : например, "1h" (поддерживаются m/h/d/w — без 'M')
        start_ms, end_ms: UNIX time в миллисекундах (UTC)
        session        : requests.Session (уже созданная снаружи)
        limit_per_req  : 1..1000 (фактический лимит Binance — 1000)
        pause_sec      : базовая пауза между запросами/повторами
        max_retries    : повторы при ошибках сети/429

    Примечание:
        Для фьючерсов USDT-M смените endpoint на 'https://fapi.binance.com/fapi/v1/klines'.
        Для COIN-M: 'https://dapi.binance.com/dapi/v1/klines'.
    """
    import time
    import math
    import random
    import requests

    def _interval_to_ms(iv: str) -> int:
        """Преобразует строковый интервал Binance в миллисекунды (поддержка m/h/d/w)."""
        if not iv or len(iv) < 2:
            raise ValueError(f"Некорректный interval: {iv!r}")
        num_part = iv[:-1]
        unit = iv[-1].lower()
        try:
            n = int(num_part)
        except Exception:
            raise ValueError(f"Некорректный interval: {iv!r}")
        if n <= 0:
            raise ValueError(f"Некорректный interval: {iv!r}")

        if unit == 'm':
            return n * 60_000
        if unit == 'h':
            return n * 3_600_000
        if unit == 'd':
            return n * 86_400_000
        if unit == 'w':
            return n * 604_800_000

        raise ValueError(f"Неподдерживаемый интервал: {iv!r} (поддерживаются m/h/d/w)")

    # --- валидации входа ---
    if limit_per_req < 1:
        limit_per_req = 1
    if limit_per_req > 1000:
        limit_per_req = 1000
    if start_ms > end_ms:
        return []

    interval_ms = _interval_to_ms(interval)
    endpoint = "https://api.binance.com/api/v3/klines"  # SPOT
    all_rows = []
    cursor = int(start_ms)

    while cursor <= end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": limit_per_req,
        }

        # повторы при сетевых ошибках/429
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = session.get(endpoint, params=params, timeout=15)
                # обработка rate-limit
                if resp.status_code in (418, 429):
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        # сервер может прислать секунды
                        try:
                            sleep_s = float(retry_after)
                        except Exception:
                            sleep_s = pause_sec * (2 ** (attempt - 1))
                    else:
                        sleep_s = pause_sec * (2 ** (attempt - 1))
                    time.sleep(sleep_s)
                    if attempt >= max_retries:
                        resp.raise_for_status()  # даст исключение
                    continue

                resp.raise_for_status()
                data = resp.json()
                if not isinstance(data, list):
                    raise ValueError(f"Неожиданный ответ Binance: {data!r}")

                # пустой ответ — дальше данных нет/окно исчерпано
                if not data:
                    cursor = end_ms + 1
                    break

                all_rows.extend(data)

                # продвижение курсора по последней свече
                last_open = int(data[-1][0])
                next_cursor = last_open + interval_ms

                # защита от зацикливания
                if next_cursor <= cursor:
                    next_cursor = cursor + interval_ms

                cursor = next_cursor

                # пауза между запросами чтобы быть бережными к API
                if cursor <= end_ms:
                    time.sleep(pause_sec)
                break

            except (requests.RequestException, ValueError) as e:
                if attempt >= max_retries:
                    raise
                # экспоненциальный бэкофф с небольшим джиттером
                sleep_s = pause_sec * (2 ** (attempt - 1)) * (1.0 + 0.25 * random.random())
                time.sleep(sleep_s)
                continue

    # дедуп + фильтрация по окну и сортировка по openTime
    seen = set()
    deduped = []
    for row in all_rows:
        # у SPOT формат всегда list, 0-й — openTime
        try:
            ot = int(row[0])
        except Exception:
            # на редкий случай, если придёт dict
            ot = int(row.get("openTime", row.get("open_time")))
        if ot < start_ms or ot > end_ms:
            # оставляем только то, что попадает в окно
            continue
        if ot in seen:
            continue
        seen.add(ot)
        deduped.append(row)

    deduped.sort(key=lambda r: int(r[0]) if isinstance(r, (list, tuple)) else int(r.get("openTime", r.get("open_time"))))
    return deduped


def fetch_klines_for_period(
    symbol: str,
    start_t: datetime,
    end_t: datetime,
    interval: str,
    session: requests.Session,
) -> List[Dict[str, Any]]:
    """
    Качаем 1h-свечи Binance на интервал [start_t, end_t] включительно по open_time.
    Для надёжности end_ms берём до конца последнего часа: end_t + 1h - 1ms.
    Возвращает список свечей в унифицированном словарном формате.
    """
    start_ms = to_ms(start_t)
    # включительно последнюю свечу (чей open_time == end_t)
    end_ms_exclusive = to_ms(end_t + timedelta(hours=1)) - 1

    raw = fetch_binance_klines(
        symbol=symbol,
        interval=interval,
        start_ms=start_ms,
        end_ms=end_ms_exclusive,
        session=session,
        limit_per_req=1000,
        pause_sec=0.5,
        max_retries=5,
    )
    return [kline_to_dict(k) for k in raw]

def align_metrics_with_klines(
    metrics: List[Dict[str, Any]],
    klines: List[Dict[str, Any]],
    *,
    strict: bool = True,
) -> List[Tuple[int, Dict[str, Any], Dict[str, Any]]]:
    """
    Сопоставляет каждую запись metrics по 'time' с часовой свечой по её open_time.
    Возвращает список кортежей (i, metric_dict, kline_dict) с НУЛЕВОЙ НУМЕРАЦИЕЙ i БЕЗ ПРОПУСКОВ.
    Если strict=True и свеча не найдена — бросает исключение.
    Если strict=False — пропускает отсутствующие соответствия с предупреждением.
    """
    # 1) Отсортируем metrics по времени (без изменения исходного порядка)
    with_idx: List[Tuple[int, Dict[str, Any], datetime]] = []
    for i, m in enumerate(metrics):
        t = parse_utc_time(m["time"])
        # нормализуем до начала часа на случай, если строка не ровно на 00:00
        t_hour = t.replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        with_idx.append((i, m, t_hour))
    with_idx.sort(key=lambda x: x[2])

    # 2) Построим индекс свечей по open_time (ms)
    by_open_ms: Dict[int, Dict[str, Any]] = {}
    for k in klines:
        by_open_ms[k["open_time"]] = k

    # 3) Выравнивание (индекс i делаем плотным 0..N-1 по найденным совпадениям)
    aligned: List[Tuple[int, Dict[str, Any], Dict[str, Any]]] = []
    dense_i = 0
    for _, m, t_hour in with_idx:
        key_ms = to_ms(t_hour)
        k = by_open_ms.get(key_ms)
        if not k:
            msg = f"Нет свечи для времени {t_hour.isoformat()} (ms={key_ms})."
            if strict:
                raise KeyError(msg)
            logging.warning(msg + " Пропускаю запись метрик.")
            continue
        aligned.append((dense_i, m, k))
        dense_i += 1

    return aligned

# -----------------------------
# Контекст для произвольного доступа
# -----------------------------

class AlignedContext:
    """
    Контекст, предоставляющий произвольный доступ к сопоставленным данным.
    Элементы — кортежи (i, m, k), где i — плотный индекс 0..N-1.
    """
    def __init__(self, aligned: List[Tuple[int, Dict[str, Any], Dict[str, Any]]]):
        self._aligned = aligned

    def size(self) -> int:
        return len(self._aligned)

    def get(self, j: int) -> Optional[Tuple[int, Dict[str, Any], Dict[str, Any]]]:
        """Абсолютный доступ: вернуть (j, m, k) или None, если j вне диапазона."""
        if 0 <= j < len(self._aligned):
            return self._aligned[j]
        return None

    def get_offset(self, i: int, delta: int) -> Optional[Tuple[int, Dict[str, Any], Dict[str, Any]]]:
        """Относительный доступ: вернуть элемент с индексом i+delta или None."""
        j = i + int(delta)
        return self.get(j)

    def window(self, i: int, *, before: int = 0, after: int = 0) -> List[Tuple[int, Dict[str, Any], Dict[str, Any]]]:
        """
        Вернуть окно вокруг i: [i-before, ..., i, ..., i+after], только существующие.
        """
        lo = max(0, i - max(0, int(before)))
        hi = min(len(self._aligned) - 1, i + max(0, int(after)))
        return [self._aligned[j] for j in range(lo, hi + 1)]

    def all(self) -> List[Tuple[int, Dict[str, Any], Dict[str, Any]]]:
        """Полный список (без копирования ссылок на m/k). Используйте аккуратно."""
        return self._aligned

def iter_enumerated_metrics_with_klines(
    metrics: List[Dict[str, Any]],
    symbol: str,
    session: Optional[requests.Session] = None,
    *,
    strict: bool = True,
) -> Iterator[Tuple[int, Dict[str, Any], Dict[str, Any], AlignedContext]]:
    """
    Высокоуровневый генератор: находит границы времени, качает свечи, выравнивает,
    и возвращает (i, metric, kline, ctx), где i = 0..N-1 по возрастанию времени.
    'ctx' позволяет на любой итерации обратиться к произвольным соседним и любым другим элементам:
        - ctx.get(i+1), ctx.get_offset(i, +1), ctx.window(i, before=2, after=3), ctx.size(), ctx.all()
    """
    min_t, max_t, is_mono = compute_time_bounds(metrics)
    if not is_mono:
        logging.info("Входные метрики не монотонны по времени — будут отсортированы по времени для прохода.")

    own_session = False
    if session is None:
        session = requests.Session()
        own_session = True

    try:
        kl = fetch_klines_for_period(symbol, start_t=min_t, end_t=max_t, interval="1h", session=session)
        aligned = align_metrics_with_klines(metrics, kl, strict=strict)
        ctx = AlignedContext(aligned)
        for i, m, k in aligned:
            yield i, m, k, ctx
    finally:
        if own_session:
            session.close()
            
def iter_both(cur: Dict[str, Any], prev: Dict[str, Any]):
    # dict_keys поддерживает | начиная с Python 3.9
    for key in (cur.keys() | prev.keys()):
        yield key, cur.get(key), prev.get(key)

def find_kline_index(klines, ts_ms: int) -> int:
    """
    Возвращает индекс свечи в списке klines, у которой open_time == ts_ms.
    Если не найдено, возвращает -1.
    """
    for i, kline in enumerate(klines):
        if kline[0] == ts_ms:
            return i
    return -1
    
def main():
    metrics = load_compact_metrics('metrics.json')
    mn, mx, mono = compute_time_bounds(metrics)
    print(f"Раннее время:  {mn.isoformat()}  (ms={to_ms(mn)})")
    print(f"Позднее время: {mx.isoformat()}  (ms={to_ms(mx)})")
    print(f"Монотонно по времени: {mono}")
    
    # dt = datetime.strptime(metrics[0]['time'], "%y-%m-%d %H:%M:%S")
    # start_time = dt.replace(tzinfo=timezone.utc).timestamp()*1000
    # dt = datetime.strptime(metrics[-1]['time'], "%y-%m-%d %H:%M:%S")
    # end_time = dt.replace(tzinfo=timezone.utc).timestamp()*1000  # указываем что это UTC
    # session = requests.Session()
    # klines30 = fetch_binance_klines(symbol='BTCUSDT', interval="30m", start_ms=int(start_time), end_ms=int(end_time), session=session)
    
    with requests.Session() as s:
        for i, m, k, ctx in iter_enumerated_metrics_with_klines(metrics, symbol="BTCUSDT", session=s, strict=False):
            # i — индекс в порядке времени (0..), m — ваш словарь метрик, k — словарь со свечой
            # ctx — произвольный доступ к любым данным: ctx.get(i+1), ctx.get_offset(i, -2), ctx.window(i, before=1, after=2), ...
            
            # dt = datetime.strptime(m['time'], "%y-%m-%d %H:%M:%S")
            # dt = dt.replace(tzinfo=timezone.utc).timestamp()*1000  # указываем что это UTC
            # ind = find_kline_index(klines30, dt)

            if i > 0:
                profit = (k['close']-k['open']) / k['open']

                pi, pm, pk = ctx.get_offset(i, -1)
                sv.metrics['dir'] = (pk['close']-pk['open'])/pk['open']
                
                for d, v_now, v_prev in iter_both(m, pm):
                    if d == 'time':
                        continue
                    sv.metrics[f'{d}_trend'] = v_now-v_prev
                    sv.metrics[d] = v_now

                report = {
                    'open_time': k["open_time"],
                    'close_time': k["close_time"],
                    'type_of_signal': 1,
                    'type_of_close': 'time_close',
                    'profit': profit,
                    'duration': 60,
                    'metrics': copy.copy(sv.metrics)
                }
                sv.positions_list.append(copy.copy(report))
    
    # print(sv.positions_list[0])
    bundle = tr.train_best_model(sv.positions_list, test_size=0.25, n_splits=5, tx_cost=0.0)
    joblib.dump(bundle, "model_bundle.joblib", compress=0)
    print("=== Holdout report ===")
    for k, v in bundle.holdout_report.items():
        if k != "oof_auc_selected":
            print(f"{k}: {v}")
    print("OOF metrics (selected):", bundle.holdout_report["oof_auc_selected"])
    print("Confusion matrix [ [TN, FP], [FN, TP] ]:\n", bundle.holdout_confusion)
    print("Top perm importance:\n", bundle.perm_importance.head(10))
    print("Top pair synergy:\n", bundle.pair_synergy.head(10))

    ex = sv.positions_list[-1]["metrics"]
    print(ex)
    print("Proba:", tr.predict_proba(bundle, ex))
    print("Label (pnl):", tr.predict_label(bundle, ex, mode="pnl"))
    # print("Example label:", tr.predict_label(model, example_metrics))
            
    # viz.plot_profit_bars_with_stats(sv.positions_list, out_dir="_viz_statistic")
    # st = stat.compute_trading_stats(sv.positions_list)
    # stat.plot_stats_overview(st, out_dir="_viz_statistic", filename="overview_stats.png")
    # cor.analyze_trades(sv.positions_list, metric_keys=list(sv.metrics.keys()))
    
if __name__ == "__main__":
    main()
