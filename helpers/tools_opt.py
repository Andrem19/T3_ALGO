import sys
from datetime import datetime, timezone
import math

def get_p_otm(snap_params: list, otype: str, opt_mon: str):
    for snap in snap_params:
        if snap['option_type'] == otype and snap['opt_mon'] in opt_mon:
            return snap
    return None

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
    
def iv_to_q(iv_annual: float, hours_left: float):
    """
    Пересчёт годовой IV в 'типичный ход' (q-единицу) за заданное количество часов.

    :param iv_annual: Годовая implied volatility, например 0.2892 (28.92%)
    :param hours_left: Кол-во часов до экспирации
    :return: (ход_в_долях, ход_в_процентах)
    """
    if iv_annual <= 0 or hours_left <= 0:
        return 0.0, 0.0

    # Переводим часы в долю года (берём 365 дней)
    year_fraction = hours_left / (365 * 24)

    # Типичный ход = годовая IV * корень из доли года
    q_fraction = iv_annual * math.sqrt(year_fraction)

    return q_fraction

def get_over(snapshot_spot_open, strike, opt_type):
    """
    Вычисляет относительное отклонение цены от страйка для опциона.
    
    snapshot_spot_open : float  # Текущая цена базового актива
    strike              : float  # Страйк опциона
    opt_type            : str    # 'C' или 'P'
    
    Возвращает: float в долях (например, 0.05 = 5%)
    """
    if snapshot_spot_open <= 0:
        raise ValueError("spot price must be positive")
    if opt_type not in ('C', 'P'):
        raise ValueError("opt_type must be 'C' or 'P'")

    # Если цены равны — ATM
    if strike == snapshot_spot_open:
        return 0.0

    # Для Call
    if opt_type == 'C':
        return abs(snapshot_spot_open - strike) / snapshot_spot_open

    # Для Put
    elif opt_type == 'P':
        return abs(snapshot_spot_open - strike) / snapshot_spot_open

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

def fit_best(strike, cur_price, askPrice):
    
    sl = [0.01, 0.012, 0.014, 0.016, 0.018, 0.02]
    tp = [0.01, 0.012, 0.014, 0.016, 0.018, 0.02]
    
    qty_set = [0.014, 0.016, 0.018, 0.02, 0.022, 0.024, 0.026, 0.028, 0.030]
    
    opt_qty = [0.014, 0.016, 0.018, 0.02, 0.022, 0.024, 0.026, 0.028, 0.030]
    
    best_qty = 0
    best_sl = 0
    best_tp = 0
    
    best_pnl = -sys.float_info.max
    pnl_up = 0
    pnl_down = 0
    best_opt_q = 0
    
    for s in sl:
        for t in tp:
            for q in qty_set:
                for oq in opt_qty:
                    sl_price = cur_price*(1-s)
                    if sl_price > strike:
                        continue
                    profit_tp = ((cur_price*(1+t))-cur_price)*q
                    
                    profit_sl = (sl_price-cur_price)*q
                    profit_opt = ((strike -sl_price)*oq)-(askPrice*oq)
                    
                    final_up = profit_tp-(askPrice*oq)
                    final_down = profit_sl+profit_opt

                    min_best_result = min(final_down, final_up)
                    if min_best_result > best_pnl:
                        best_pnl = min_best_result
                        best_tp = t
                        best_sl = s
                        best_qty = q
                        best_opt_q = oq
                        
                        pnl_up = final_up
                        pnl_down = final_down
    if best_pnl < 0:
        return None
    
    return {
        'pnl_up': pnl_up,
        'pnl_down': pnl_down,
        'best_qty': best_qty,
        'best_sl': best_sl,
        'best_tp': best_tp,
        'best_pnl': best_pnl,
        'best_opt_q': best_opt_q
    }