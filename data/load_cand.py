# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Dict
import os
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd


def _to_utc_ms(dt: datetime) -> int:
    """
    Преобразовать datetime в миллисекунды UNIX (UTC).
    Если dt наивный (tzinfo is None) — считаем его в UTC.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return int(dt.timestamp() * 1000)


def load_klines_numpy(
    csv_path: str,
    period_start: datetime,
    period_end: datetime,
    *,
    expand_end_if_midnight: bool = True,
) -> np.ndarray:
    """
    Прочитать kline CSV и вернуть 2D numpy.ndarray (каждая строка — свеча).
    Фильтрация по первой колонке (open time, мс, UTC).

    Параметры:
      csv_path: путь к файлу, например "_crypto_data/spot_BTCUSDT/5m/BTCUSDT-5m.csv"
      period_start: старт периода (datetime). Наивные datetime трактуются как UTC.
      period_end: конец периода (datetime). По умолчанию:
                  если время у конца ровно 00:00:00 и expand_end_if_midnight=True,
                  то конец расширяется на +1 день (чтобы включить весь день).
      expand_end_if_midnight: логика удобства для диапазонов «по датам».
                              Если True и period_end в полночь — смещаем конец на +1 день.

    Возвращает:
      numpy.ndarray формы (N, M). Если данных нет — пустой массив (0, 0) или (0, M).

    Примечание по фильтру:
      Диапазон: start_ms <= open_time_ms < end_ms_exclusive
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Файл не найден: {csv_path}")

    if expand_end_if_midnight and (
        period_end.hour == 0
        and period_end.minute == 0
        and period_end.second == 0
        and period_end.microsecond == 0
    ):
        # Удобная семантика: конец дня включительно -> делаем эксклюзивный конец на следующее утро
        period_end = period_end + timedelta(days=1)

    start_ms = _to_utc_ms(period_start)
    end_ms_exclusive = _to_utc_ms(period_end)

    if end_ms_exclusive <= start_ms:
        # Защита от перепутанных границ
        raise ValueError("period_end должен быть позже period_start")

    # Читаем как строки, затем приводим к числам (отфильтровываем возможные заголовки/мусор)
    df = pd.read_csv(csv_path, header=None, dtype=str, low_memory=False)

    # Приводим все колонки к числам; нечисловые -> NaN
    df = df.apply(pd.to_numeric, errors="coerce")
    # Удаляем строки без валидного таймстампа в первой колонке
    df = df[df[0].notna()].copy()

    if df.empty:
        return np.empty((0, 0), dtype=float)

    # Фильтр по открытию свечи (первая колонка, мс)
    # Важно: после to_numeric колонка 0 float64. Для корректного сравнения — без округления.
    mask = (df[0] >= float(start_ms)) & (df[0] < float(end_ms_exclusive))
    df = df.loc[mask]

    if df.empty:
        return np.empty((0, df.shape[1]), dtype=float)

    # Возвращаем как плотный numpy-массив (числовой). Времена будут в float64,
    # что для значений ~1.7e12 (мс) сохраняет точность в float64.
    return df.to_numpy()




def load_enriched_klines(
    csv_path: str,
    start: datetime,
    end: datetime,
) -> List[Dict]:
    """
    Загружает обогащённые свечи из csv_path за [start, end).
    Требования: файл уже содержит header (как мы его сохранили при обогащении).

    Возвращает list[dict], где ключи = имена колонок из хедера.
    Времена трактуются как UTC; если start/end naive — считаем их UTC.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    # Нормализуем в UTC и переводим в миллисекунды
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    else:
        start = start.astimezone(timezone.utc)

    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    else:
        end = end.astimezone(timezone.utc)

    start_ms = int(start.timestamp() * 1000)
    end_ms   = int(end.timestamp() * 1000)

    # Читаем с хедером; проверим, что колонка open_time есть
    head = pd.read_csv(csv_path, nrows=0)
    if "open_time" not in head.columns:
        raise ValueError(
            f"Файл {csv_path} не содержит ожидаемый header с колонкой 'open_time'. "
            "Убедитесь, что вы предварительно прогнали обогащение."
        )

    df = pd.read_csv(csv_path)
    # Фильтр по диапазону (включительно по start, эксклюзивно по end)
    df = df[(df["open_time"] >= start_ms) & (df["open_time"] < end_ms)].copy()
    df.sort_values("open_time", inplace=True)

    # Приведём numpy-значения к «обычным» питоновским типам
    records = []
    for row in df.to_dict(orient="records"):
        clean = {}
        for k, v in row.items():
            if pd.isna(v):
                clean[k] = None
            else:
                try:
                    clean[k] = v.item()  # numpy → python scalar
                except Exception:
                    clean[k] = v
        records.append(clean)
    return records

# пример использования:
# fut_path = '/home/jupiter/PYTHON/MARKET_DATA/_crypto_data/fut_BTCUSDT/1h/BTCUSDT-1h.csv'
# from datetime import datetime
# data = load_enriched_klines(fut_path, datetime(2020,1,1), datetime(2025,1,1))
# print(len(data), data[0])
