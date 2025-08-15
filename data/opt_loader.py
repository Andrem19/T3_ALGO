#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split huge Deribit options snapshots CSV into monthly files + lazy monthly loader (NO pandas).

ЧАСТЬ 1: split_big_csv_by_month()
- Берёт большой CSV (формат как в generator.py) и записывает помесячные файлы:
  <OUT_MONTH_DIR>/<CURRENCY>_snapshots_1m_YYYY-MM.csv
- Создаёт манифест JSON: <OUT_MONTH_DIR>/manifest.json
  {
    "BTC": {
      "2020-01": {"path": ".../BTC_snapshots_1m_2020-01.csv",
                  "start_ms": 1577836800000, "end_ms": 1580515199999},
      ...
    },
    "ETH": { ... }
  }

ЧАСТЬ 2: MonthlySnapshotLoader
- Держит в RAM только текущий месяц.
- Метод get_snapshot(ts_ms) возвращает список опционов для минутного таймстампа.
- При уходе в другой месяц выгружает старый и грузит новый.

Формат CSV — как у нашего generator.py:
minute_ts_ms,currency,instrument_name,option_type,strike,expiry,snapshot_spot_open,
volume_contracts,notional_btc,vwap_btc,last_trade_price_btc,last_iv_pct,last_mark_btc,
last_direction,last_trade_ts_ms,trade_count,buy_volume_contracts,buy_notional_btc,
sell_volume_contracts,sell_notional_btc,theo_usd,theo_btc,delta,gamma,vega,theta,rho,
bid_est_usd,ask_est_usd,mid_est_usd,fill_gap_ms
"""

from __future__ import annotations

# ======================= НАСТРОЙКИ =======================


# ======================= ИМПОРТЫ =======================
import csv
import json
import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from helpers.tools_opt import classify_option, hours_to_expiry
from shared_vars import OUT_MONTH_DIR, SRC_SNAPSHOT_CSV, FILENAME_TEMPLATE, MANIFEST_PATH

# ======================= УТИЛИТЫ =======================
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _to_int(x: str) -> Optional[int]:
    if x is None or x == "":
        return None
    try:
        return int(float(x))
    except Exception:
        return None

def _to_float(x: str) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None

def _ym_from_ms(t_ms: int) -> (int, int, str):
    dt = datetime.fromtimestamp(t_ms / 1000.0, tz=timezone.utc)
    return dt.year, dt.month, f"{dt.year:04d}-{dt.month:02d}"

def _detect_currency_from_basename(path: str) -> Optional[str]:
    """
    Пытаемся извлечь 'BTC' / 'ETH' из имени файла, например 'BTC_snapshots_1m.csv'.
    Возвращает 'BTC' / 'ETH' или None.
    """
    base = os.path.basename(path)
    m = re.match(r'^([A-Z]{3,5})_snapshots_1m(?:_|\.|$)', base)
    return m.group(1) if m else None

# ======================= 1) СПЛИТ НА МЕСЯЦЫ =======================
def split_big_csv_by_month(
    src_csv: str = SRC_SNAPSHOT_CSV,
    out_dir: str = OUT_MONTH_DIR,
    manifest_path: str = MANIFEST_PATH
) -> None:
    """
    Читает большой snapshots-CSV и записывает помесячные файлы, создаёт manifest.json.
    Имена: <out_dir>/<CURRENCY>_snapshots_1m_YYYY-MM.csv
    """
    _ensure_dir(out_dir)
    _ensure_dir(os.path.dirname(manifest_path))

    # currency по умолчанию — из имени файла (если получится)
    currency_from_name = _detect_currency_from_basename(src_csv)

    # держим открытые файловые дескрипторы для месяцев
    writers: Dict[str, csv.writer] = {}
    files: Dict[str, Any] = {}
    # учёт min/max по месяцам и по currency
    manifest: Dict[str, Dict[str, Dict[str, Any]]] = {}

    with open(src_csv, "r", encoding="utf-8") as fin:
        rd = csv.reader(fin)
        header = next(rd, None)
        if not header:
            print("[split] Пустой входной файл.")
            return

        # индексы колонок
        idx = {name: i for i, name in enumerate(header)}
        def _get(name: str) -> int:
            if name not in idx:
                raise KeyError(f"[split] Ожидаемая колонка отсутствует: {name}")
            return idx[name]

        col_minute_ts_ms = _get("minute_ts_ms")
        col_currency     = idx.get("currency", None)  # на всякий случай

        for row in rd:
            if not row:
                continue

            t = _to_int(row[col_minute_ts_ms])
            if t is None:
                continue

            # определяем валюту строки
            cur = (row[col_currency] if col_currency is not None and row[col_currency] else currency_from_name) or "UNK"

            y, m, ym = _ym_from_ms(t)
            cur_dir = out_dir  # можно делать иерархию по годам, если захотите
            _ensure_dir(cur_dir)

            # имя помесячного файла
            fname = FILENAME_TEMPLATE.format(currency=cur, year=y, month=m)
            path_month = os.path.join(cur_dir, fname)

            # если для этого месяца ещё не создан writer — создаём и пишем header
            if ym not in writers:
                f = open(path_month, "w", encoding="utf-8", newline="")
                w = csv.writer(f)
                w.writerow(header)
                writers[ym] = w
                files[ym] = f

            # пишем строку
            writers[ym].writerow(row)

            # манифест
            if cur not in manifest:
                manifest[cur] = {}
            mm = manifest[cur].setdefault(ym, {"path": path_month, "start_ms": t, "end_ms": t})
            if t < mm["start_ms"]:
                mm["start_ms"] = t
            if t > mm["end_ms"]:
                mm["end_ms"] = t

    # закрываем все файлы
    for ym, f in files.items():
        try:
            f.flush()
            f.close()
        except Exception:
            pass

    # пишем manifest.json
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, ensure_ascii=False, indent=2)

    print(f"[split] Готово. Манифест: {manifest_path}")
    print(f"[split] Папка с помесячными файлами: {out_dir}")

# ======================= 2) ЛОАДЕР МЕСЯЦЕВ =======================
class MonthlySnapshotLoader:
    """
    Ленивая загрузка помесячных snapshots CSV.
    Держит в памяти только один текущий месяц. При запросе минуты из другого месяца —
    грузит его и выгружает прошлый.

    Использование:
        loader = MonthlySnapshotLoader(month_dir=OUT_MONTH_DIR, manifest_path=MANIFEST_PATH, currency="BTC")
        recs = loader.get_snapshot(ts_ms)  # список dict с полями строки из snapshots CSV
    """
    def __init__(self,
                 month_dir: str = OUT_MONTH_DIR,
                 manifest_path: str = MANIFEST_PATH,
                 currency: str = "BTC"):
        self.month_dir = month_dir
        self.manifest_path = manifest_path
        self.currency = currency

        self.manifest = self._load_or_build_manifest()
        if self.currency not in self.manifest:
            raise RuntimeError(f"[loader] В манифесте нет валюты {self.currency}. "
                               f"Путь манифеста: {self.manifest_path}")

        self.current_month_key: Optional[str] = None
        self.current_month_dict: Optional[Dict[int, List[dict]]] = None

    # ---------- внутренние утилиты ----------
    def _load_or_build_manifest(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Загружает manifest.json; если его нет — построит из файлов в директории.
        Ожидаемые файлы: <CURRENCY>_snapshots_1m_YYYY-MM.csv
        """
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)

        # сканируем директорию и собираем простой манифест без точных границ
        print("[loader] manifest.json не найден — строю из файлов директории.")
        manifest: Dict[str, Dict[str, Dict[str, Any]]] = {}
        pat = re.compile(r'^([A-Z]{3,5})_snapshots_1m_(\d{4})-(\d{2})\.csv$')
        for name in os.listdir(self.month_dir):
            m = pat.match(name)
            if not m:
                continue
            cur = m.group(1)
            y = int(m.group(2))
            mo = int(m.group(3))
            ym = f"{y:04d}-{mo:02d}"
            path = os.path.join(self.month_dir, name)
            if cur not in manifest:
                manifest[cur] = {}
            manifest[cur][ym] = {"path": path, "start_ms": None, "end_ms": None}

        # сохраняем
        _ensure_dir(os.path.dirname(self.manifest_path))
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        return manifest

    def _month_key(self, ts_ms: int) -> str:
        return _ym_from_ms(ts_ms)[2]

    def _path_for_month(self, ym: str) -> Optional[str]:
        rec = self.manifest.get(self.currency, {}).get(ym)
        return rec["path"] if rec else None

    # ---------- основная загрузка месяца ----------
    def _load_month_into_memory(self, ym: str) -> Dict[int, List[dict]]:
        """
        Читает один помесячный CSV и собирает dict: minute_ts_ms -> list[dict].
        """
        path = self._path_for_month(ym)
        if not path or not os.path.exists(path):
            # файл для этого месяца отсутствует
            return {}

        out: Dict[int, List[dict]] = {}
        with open(path, "r", encoding="utf-8") as f:
            rd = csv.reader(f)
            header = next(rd, None)
            if not header:
                return out
            idx = {name: i for i, name in enumerate(header)}
            # обязательные поля:
            need = [
                "minute_ts_ms","currency","instrument_name","option_type","strike","expiry",
                "snapshot_spot_open","volume_contracts","notional_btc","vwap_btc",
                "last_trade_price_btc","last_iv_pct","last_mark_btc","last_direction","last_trade_ts_ms",
                "trade_count","buy_volume_contracts","buy_notional_btc","sell_volume_contracts","sell_notional_btc",
                "theo_usd","theo_btc","delta","gamma","vega","theta","rho",
                "bid_est_usd","ask_est_usd","mid_est_usd"
            ]
            for n in need:
                if n not in idx:
                    raise RuntimeError(f"[loader] В файле {path} отсутствует требуемая колонка: {n}")
            # опциональная:
            fill_idx = idx.get("fill_gap_ms", None)

            for row in rd:
                if not row:
                    continue
                t = _to_int(row[idx["minute_ts_ms"]])
                if t is None:
                    continue
                rec = {
                    "minute_ts_ms": t,
                    "currency": row[idx["currency"]],
                    "instrument_name": row[idx["instrument_name"]],
                    "option_type": row[idx["option_type"]],
                    "strike": float(row[idx["strike"]]) if row[idx["strike"]] else 0.0,
                    "expiry": row[idx["expiry"]],
                    "snapshot_spot_open": float(row[idx["snapshot_spot_open"]]) if row[idx["snapshot_spot_open"]] else 0.0,
                    "volume_contracts": float(row[idx["volume_contracts"]]) if row[idx["volume_contracts"]] else 0.0,
                    "notional_btc": float(row[idx["notional_btc"]]) if row[idx["notional_btc"]] else 0.0,
                    "vwap_btc": _to_float(row[idx["vwap_btc"]]),
                    "last_trade_price_btc": _to_float(row[idx["last_trade_price_btc"]]),
                    "last_iv_pct": _to_float(row[idx["last_iv_pct"]]),
                    "last_mark_btc": _to_float(row[idx["last_mark_btc"]]),
                    "last_direction": row[idx["last_direction"]] or None,
                    "last_trade_ts_ms": _to_int(row[idx["last_trade_ts_ms"]]),
                    "trade_count": int(float(row[idx["trade_count"]])) if row[idx["trade_count"]] else 0,
                    "buy_volume_contracts": float(row[idx["buy_volume_contracts"]]) if row[idx["buy_volume_contracts"]] else 0.0,
                    "buy_notional_btc": float(row[idx["buy_notional_btc"]]) if row[idx["buy_notional_btc"]] else 0.0,
                    "sell_volume_contracts": float(row[idx["sell_volume_contracts"]]) if row[idx["sell_volume_contracts"]] else 0.0,
                    "sell_notional_btc": float(row[idx["sell_notional_btc"]]) if row[idx["sell_notional_btc"]] else 0.0,
                    "theo_usd": _to_float(row[idx["theo_usd"]]),
                    "theo_btc": _to_float(row[idx["theo_btc"]]),
                    "delta": _to_float(row[idx["delta"]]),
                    "gamma": _to_float(row[idx["gamma"]]),
                    "vega": _to_float(row[idx["vega"]]),
                    "theta": _to_float(row[idx["theta"]]),
                    "rho": _to_float(row[idx["rho"]]),
                    "bid_est_usd": _to_float(row[idx["bid_est_usd"]]),
                    "askPrice": _to_float(row[idx["ask_est_usd"]]),
                    "mid_est_usd": _to_float(row[idx["mid_est_usd"]]),
                    "fill_gap_ms": (_to_int(row[fill_idx]) if fill_idx is not None and row[fill_idx] != "" else None),
                }
                rec['opt_mon'] = classify_option(rec)
                rec['hours_to_exp'] = hours_to_expiry(row[idx["expiry"]], t)
                out.setdefault(t, []).append(rec)
        return out

    def classify_option(option):
        S = option["snapshot_spot_open"]
        K = option["strike"]
        t = option["option_type"]

        # ATM критерий (например, ±0.5%)
        if abs(K - S) / S <= 0.005:
            return "ATM"

        if t == "C":
            return "ITM" if S > K else "OTM"
        elif t == "P":
            return "ITM" if S < K else "OTM"
        else:
            return None
    # ---------- публичные методы ----------
    def get_snapshot(self, ts_ms: int) -> List[dict]:
        """
        Возвращает список опционов (снимок) для минутного таймстампа ts_ms.
        При необходимости лениво подгружает нужный месяц и выгружает предыдущий.
        """
        ym = self._month_key(ts_ms)
        if ym != self.current_month_key:
            # выгрузить старый
            self.current_month_dict = None
            # загрузить новый
            self.current_month_dict = self._load_month_into_memory(ym)
            self.current_month_key = ym
        if not self.current_month_dict:
            return []
        return self.current_month_dict.get(ts_ms, [])
    
    def hours_to_expiry(expiry: str, current_ts_ms: int) -> float:
        """
        Считает количество часов до экспирации опциона.
        
        :param expiry: дата экспирации в формате 'YYYY-MM-DD'
        :param current_ts_ms: текущий момент времени в миллисекундах (UTC)
        :return: оставшееся время до экспирации в часах (float)
        """
        # Конвертируем строку даты в datetime и задаём время 08:00 UTC
        expiry_dt = datetime.strptime(expiry, "%Y-%m-%d").replace(
            hour=8, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
        )
        
        # Текущая дата-время из миллисекунд
        now_dt = datetime.fromtimestamp(current_ts_ms / 1000, tz=timezone.utc)
        
        # Разница во времени
        diff = expiry_dt - now_dt
        return diff.total_seconds() / 3600


    def get_snapshot_filtered(self,
                            ts_ms: int,
                            *,
                            min_hours_to_expiry: Optional[float] = None,
                            max_hours_to_expiry: Optional[float] = None,
                            opt_type: Optional[str] = None,
                            max_moneyness_pct: Optional[float] = None,
                            expiry_hour_utc: int = 8) -> List[dict]:
        """
        Возвращает отфильтрованный список опционов для минутного таймстампа ts_ms.

        Параметры:
        - min_hours_to_expiry: если задано — оставить только опционы с T_expiry - t >= min (часы).
        - max_hours_to_expiry: если задано — оставить только опционы с T_expiry - t <= max (часы).
        - opt_type: 'C', 'P' или None (оба).
        - max_moneyness_pct: |K - S| / S <= значение (например, 0.02 = ±2%).
        - expiry_hour_utc: час UTC окончания для даты expiry (обычно 08:00 для Deribit).

        Сортировка результата — по близости страйка к споту.
        """
        recs = self.get_snapshot(ts_ms)
        if not recs:
            return []

        t_dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        spot = recs[0].get("snapshot_spot_open", 0.0) or 0.0

        out: List[Dict] = []
        for r in recs:
            # фильтр по типу
            if opt_type in ("C", "P") and r.get("option_type") != opt_type:
                continue

            # фильтр по moneyness
            if max_moneyness_pct is not None and spot > 0:
                K = float(r.get("strike", 0.0) or 0.0)
                if abs(K - spot) / spot > max_moneyness_pct:
                    continue

            # фильтр по времени до экспирации
            if (min_hours_to_expiry is not None) or (max_hours_to_expiry is not None):
                exp_str = r.get("expiry") or ""  # 'YYYY-MM-DD'
                try:
                    y, m, d = map(int, exp_str.split("-"))
                    exp_dt = datetime(y, m, d, hour=expiry_hour_utc, tzinfo=timezone.utc)
                    hours_left = (exp_dt - t_dt).total_seconds() / 3600.0
                except Exception:
                    # Если не удалось разобрать дату — пропускаем запись.
                    continue

                if (min_hours_to_expiry is not None) and (hours_left < float(min_hours_to_expiry)):
                    continue
                if (max_hours_to_expiry is not None) and (hours_left > float(max_hours_to_expiry)):
                    continue

            out.append(r)

        if spot > 0:
            out.sort(key=lambda x: abs(float(x.get("strike", 0.0) or 0.0) - spot))
        return out


# ======================= ПРИМЕР ИСПОЛЬЗОВАНИЯ =======================
# if __name__ == "__main__":
    # 1) Разбивка огромного файла на месяцы (выполняется один раз):
    #    Убедитесь, что пути в константах верные. Раскомментируйте, запустите, затем закомментируйте обратно.
    # split_big_csv_by_month(SRC_SNAPSHOT_CSV, OUT_MONTH_DIR, MANIFEST_PATH)

    # # 2) Работа через ленивый загрузчик:
    # #    Пример — берём минуту и печатаем количество опционов в этой минуте.
    # loader = MonthlySnapshotLoader(month_dir=OUT_MONTH_DIR, manifest_path=MANIFEST_PATH, currency="BTC")

    # # пример: 2023-01-01T00:00:00Z
    # ts_example = int(datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
    # snap = loader.get_snapshot(ts_example)
    # print(f"Minute {ts_example} options: {len(snap)}")

    # # переход в другой месяц — загрузка произойдёт автоматически
    # ts_example2 = int(datetime(2023, 2, 1, 12, 34, 0, tzinfo=timezone.utc).timestamp() * 1000)
    # snap2 = loader.get_snapshot(ts_example2)
    # print(f"Minute {ts_example2} options: {len(snap2)}")
