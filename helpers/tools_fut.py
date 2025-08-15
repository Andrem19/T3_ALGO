import numpy as np
import shared_vars as sv
from numba import jit
import talib
import pandas as pd


@jit(nopython=True)
def convert_timeframe(candels: np.ndarray, timeframe: int, ln: int):
    
    opens = candels[:, 1]
    highs = candels[:, 2]
    lows = candels[:, 3]
    closes = candels[:, 4]
    
    lenth_opens = len(opens)
    length = lenth_opens // timeframe if ln == 0 else ln

    new_opens = np.zeros(length)
    new_highs = np.zeros(length)
    new_lows = np.zeros(length)
    new_closes = np.zeros(length)

    for i in range(length):
        start = lenth_opens - (i + 1) * timeframe
        end = lenth_opens - i * timeframe

        new_opens[-(i + 1)] = opens[start]
        new_highs[-(i + 1)] = np.max(highs[start:end])
        new_lows[-(i + 1)] = np.min(lows[start:end])
        new_closes[-(i + 1)] = closes[end - 1]

    return new_opens, new_highs, new_lows, new_closes

def get_tail_body(open, high, low, close):
    body = abs(open - close)
    if open < close:
        min_br = open
        max_br = close
    else:
        min_br = close
        max_br = open
    low_tail = min_br - low
    high_tail = high - max_br
    return low_tail, high_tail, body

def classify_candlestick(body: float, upper: float, lower: float) -> int:
    SMALL = 0.0025         # Threshold to consider a value negligible (0.5%)
    LONG_MULTIPLIER = 2.5   # Multiplier to decide if a shadow is "long" relative to the body

    # Convert lower shadow to positive for easier calculations
    lower = abs(lower)

    # Check for Doji: nearly zero body
    if abs(body) < SMALL:
        # Doji classification based on shadows
        if upper < SMALL and lower < SMALL:
            return 9      # Pattern 9: Doji with negligible shadows
        elif upper >= lower:
            if lower < SMALL:
                return 10  # Pattern 10: Doji with long upper shadow
            else:
                return 0  # Pattern 12: Doji with both shadows significant (upper dominant)
        else:  # lower > upper
            if upper < SMALL:
                return 11  # Pattern 11: Doji with long lower shadow
            else:
                return 0  # Pattern 12: Doji with both shadows significant (lower dominant)

    # For non-Doji candles (body is significant)
    is_bullish = body > 0
    abs_body = abs(body)
    is_upper_negligible = upper < SMALL
    is_lower_negligible = lower < SMALL
    is_upper_long = upper >= LONG_MULTIPLIER * abs_body
    is_lower_long = lower >= LONG_MULTIPLIER * abs_body

    if is_bullish:
        if is_upper_negligible and is_lower_negligible:
            return 1  # Pattern 1: Bullish Marubozu
        elif is_upper_long and is_lower_negligible:
            return 2  # Pattern 2: Bullish with long upper shadow
        elif is_lower_long and is_upper_negligible:
            return 3  # Pattern 3: Bullish with long lower shadow
        elif not is_upper_negligible and not is_lower_negligible:
            return 4  # Pattern 4: Bullish with both shadows significant
        else:
            # When one shadow exists but does not qualify as "long"
            if not is_upper_negligible:
                return 2  # Default to pattern 2 if only upper shadow is present
            elif not is_lower_negligible:
                return 3  # Default to pattern 3 if only lower shadow is present
            else:
                return 4
    else:
        # Bearish candle
        if is_upper_negligible and is_lower_negligible:
            return 5  # Pattern 5: Bearish Marubozu
        elif is_upper_long and is_lower_negligible:
            return 6  # Pattern 6: Bearish with long upper shadow
        elif is_lower_long and is_upper_negligible:
            return 7  # Pattern 7: Bearish with long lower shadow
        elif not is_upper_negligible and not is_lower_negligible:
            return 8  # Pattern 8: Bearish with both shadows significant
        else:
            if not is_upper_negligible:
                return 6
            elif not is_lower_negligible:
                return 7
            else:
                return 8


def classify_two_candlesticks(
    body_1: float, upper_1: float, lower_1: float,
    body_2: float, upper_2: float, lower_2: float
) -> int:
    """
    Классификация двух подряд идущих свечей (двухсвечный паттерн) в одно из 24 значений (1..24).

    Параметры:
        body_1, upper_1, lower_1 : float
            Параметры первой свечи:
                body_1  > 0  -> бычья свеча,
                body_1  < 0  -> медвежья свеча,
                body_1  ~ 0  -> doji (очень маленькое тело).
                upper_1      -> верхняя тень (если body>0, это процент над close, иначе над open)
                lower_1      -> нижняя тень (отрицательное число, берём abs(...) как длину)
        
        body_2, upper_2, lower_2 : float
            То же, но для второй свечи.

    Возвращает:
        int в диапазоне 1..24 – уникальный номер паттерна.

    -------------------------------
    Библиотека используемых паттернов (пример):
      1.  Bullish Engulfing
      2.  Bearish Engulfing
      3.  Bullish Harami
      4.  Bearish Harami
      5.  Piercing Pattern
      6.  Dark Cloud Cover
      7.  Tweezer Top
      8.  Tweezer Bottom
      9.  Bullish Outside
      10. Bearish Outside
      11. Bullish Inside
      12. Bearish Inside
      13. Bullish Doji Star
      14. Bearish Doji Star
      15. Matching High
      16. Matching Low
      17. Bullish Kicker
      18. Bearish Kicker
      19. Bullish Meeting Lines
      20. Bearish Meeting Lines
      21. Bullish Belt Hold
      22. Bearish Belt Hold
      23. Side-by-Side (Bullish)
      24. Side-by-Side (Bearish)
    -------------------------------

    Подход к реконструкции цены (условный):
      - Считаем, что первая свеча открылась по цене 1.0
      - Если body > 0: close_1 = 1.0 + body_1
        Иначе        : close_1 = 1.0 + body_1 (будет меньше 1, если body_1 < 0)
      - high_1 = max(open_1, close_1) + upper_1
      - low_1 = min(open_1, close_1) - abs(lower_1)

      Аналогично для второй свечи (open_2 = close_1) – чтобы учесть «природу» соседних свечей.
      Если нужно моделировать «гепы» (разрывы), можно open_2 = close_1 +/- некий шаг.
    """
    import math

    # === Шаг 1. Реконструкция (псевдо) цен свечей ===
    # Условимся, что первая свеча открывается по 1.0
    open_1 = 1.0
    close_1 = open_1 + body_1  # если body_1 > 0, закрытие выше открытия
    high_1 = max(open_1, close_1) + upper_1
    low_1 = min(open_1, close_1) - abs(lower_1)

    # Предположим, что вторая свеча открывается там же, где закрылась первая.
    # (В реальном рынке может быть геп: open_2 != close_1, но для упрощения возьмём так.)
    open_2 = close_1
    close_2 = open_2 + body_2
    high_2 = max(open_2, close_2) + upper_2
    low_2 = min(open_2, close_2) - abs(lower_2)

    # === Шаг 2. Определение свойств свечей (бычья / медвежья / doji) ===
    SMALL = 0.001  # порог, ниже которого тело считаем очень маленьким (doji)
    def get_candle_type(b: float) -> str:
        if abs(b) < SMALL:
            return "doji"
        return "bullish" if b > 0 else "bearish"

    candle1_type = get_candle_type(body_1)
    candle2_type = get_candle_type(body_2)

    # Размер тела (в условных единицах), чтобы оценивать "большое" или "маленькое" тело
    body1_size = abs(close_1 - open_1)
    body2_size = abs(close_2 - open_2)

    # === Шаг 3. Вспомогательные функции для сравнения ===
    def is_engulfing_bullish() -> bool:
        # Bullish Engulfing: первая свеча медвежья, вторая бычья,
        # причём тело второй полностью покрывает (engulf) тело первой
        if candle1_type == "bearish" and candle2_type == "bullish":
            return (open_2 < open_1) and (close_2 > close_1)
        return False

    def is_engulfing_bearish() -> bool:
        # Bearish Engulfing: первая свеча бычья, вторая медвежья,
        # причём тело второй полностью покрывает тело первой
        if candle1_type == "bullish" and candle2_type == "bearish":
            return (open_2 > open_1) and (close_2 < close_1)
        return False

    def is_harami_bullish() -> bool:
        # Bullish Harami: первая свеча медвежья (большая),
        # вторая (обычно бычья) внутри тела первой
        if candle1_type == "bearish" and candle2_type == "bullish":
            return (open_2 > open_1) and (close_2 < close_1)
        return False

    def is_harami_bearish() -> bool:
        # Bearish Harami: первая свеча бычья (большая),
        # вторая (обычно медвежья) внутри тела первой
        if candle1_type == "bullish" and candle2_type == "bearish":
            return (open_2 < open_1) and (close_2 > close_1)
        return False

    def is_piercing() -> bool:
        # Piercing Pattern: первая свеча медвежья,
        # вторая – бычья, открытие 2й ниже минимума 1й (геп вниз),
        # а закрытие выше середины тела 1й
        if candle1_type == "bearish" and candle2_type == "bullish":
            mid1 = open_1 + (body1_size / 2.0)  # середина тела первой
            # open_2 < low_1 (геп вниз), но у нас упрощённая логика без явных гэпов
            # Упростим до "open_2 < close_1" и "close_2 > mid тела первой"
            return (open_2 < close_1) and (close_2 > mid1)
        return False

    def is_dark_cloud() -> bool:
        # Dark Cloud Cover: первая свеча бычья,
        # вторая – медвежья, открытие 2й выше максимума 1й (геп вверх),
        # а закрытие ниже середины тела 1й
        if candle1_type == "bullish" and candle2_type == "bearish":
            mid1 = open_1 + (body1_size / 2.0)
            return (open_2 > close_1) and (close_2 < mid1)
        return False

    def is_tweezer_top() -> bool:
        # Tweezer Top: две свечи с примерно одинаковым верхом (high_1 ~ high_2)
        # Обычно первая свеча бычья, вторая – медвежья, но мы упростим до сходных high.
        return abs(high_1 - high_2) < 0.001

    def is_tweezer_bottom() -> bool:
        # Tweezer Bottom: две свечи с примерно одинаковым низом (low_1 ~ low_2)
        return abs(low_1 - low_2) < 0.001

    def is_outside_bullish() -> bool:
        # Bullish Outside: вторая свеча бычья,
        # и при этом high_2 > high_1 и low_2 < low_1 (полностью охватывает первую)
        return (candle2_type == "bullish") and (high_2 > high_1) and (low_2 < low_1)

    def is_outside_bearish() -> bool:
        return (candle2_type == "bearish") and (high_2 > high_1) and (low_2 < low_1)

    def is_inside_bullish() -> bool:
        # Bullish Inside: вторая свеча бычья и полностью внутри диапазона первой
        return (candle2_type == "bullish") and (high_2 < high_1) and (low_2 > low_1)

    def is_inside_bearish() -> bool:
        return (candle2_type == "bearish") and (high_2 < high_1) and (low_2 > low_1)

    def is_doji_star_bullish() -> bool:
        # Bullish Doji Star: первая свеча медвежья, вторая – doji, расположенная с «разрывом» (геп)
        # Упрощённо: candle2_type = doji, candle1_type = bearish
        # и high_2 < open_1 или low_2 > close_1 (в зависимости от направления) – будем считать условно
        if candle1_type == "bearish" and candle2_type == "doji":
            # Упростим логику гепа: open_2 > open_1 + некий порог
            return True
        return False

    def is_doji_star_bearish() -> bool:
        # Аналогично для первой бычьей, второй doji
        if candle1_type == "bullish" and candle2_type == "doji":
            return True
        return False

    def is_meeting_lines_bullish() -> bool:
        # Bullish Meeting Lines: первая свеча медвежья, вторая бычья
        # с открытием, равным или близким к закрытию первой.
        # И закрытие второй очень близко к закрытию первой.
        if candle1_type == "bearish" and candle2_type == "bullish":
            if abs(open_2 - close_1) < 0.001:
                return True
        return False

    def is_meeting_lines_bearish() -> bool:
        # То же зеркально
        if candle1_type == "bullish" and candle2_type == "bearish":
            if abs(open_2 - close_1) < 0.001:
                return True
        return False

    def is_belt_hold_bullish() -> bool:
        # Bullish Belt Hold: длинная бычья свеча с очень маленькой нижней тенью
        # Вторая (предположим) может быть нейтральной?
        # Упростим до: первая свеча (candle1_type == 'bullish'),
        # нижняя тень очень маленькая, а вторая просто тоже бычья.
        if candle1_type == "bullish" and abs(low_1 - min(open_1, close_1)) < 0.001:
            if candle2_type == "bullish":
                return True
        return False

    def is_belt_hold_bearish() -> bool:
        if candle1_type == "bearish" and abs(high_1 - max(open_1, close_1)) < 0.001:
            if candle2_type == "bearish":
                return True
        return False

    def is_side_by_side_bullish() -> bool:
        # Side-by-Side (Bullish): упрощённо - две бычьи свечи подряд с близкими тенями
        if candle1_type == "bullish" and candle2_type == "bullish":
            # Проверим, что high_1 ~ high_2 и/или low_1 ~ low_2
            if abs(high_1 - high_2) < 0.002 or abs(low_1 - low_2) < 0.002:
                return True
        return False

    def is_side_by_side_bearish() -> bool:
        if candle1_type == "bearish" and candle2_type == "bearish":
            if abs(high_1 - high_2) < 0.002 or abs(low_1 - low_2) < 0.002:
                return True
        return False

    # === Шаг 4. Определяем паттерн по набору проверок (упрощённый приоритет) ===
    # Для упрощения: сначала проверяем самые "узнаваемые" паттерны, далее переходим к другим.
    # На практике логику и приоритет можно менять.

    if is_engulfing_bullish():
        return 1
    if is_engulfing_bearish():
        return 2
    if is_harami_bullish():
        return 3
    if is_harami_bearish():
        return 4
    if is_piercing():
        return 5
    if is_dark_cloud():
        return 6
    if is_tweezer_top():
        return 7
    if is_tweezer_bottom():
        return 8
    if is_outside_bullish():
        return 9
    if is_outside_bearish():
        return 10
    if is_inside_bullish():
        return 11
    if is_inside_bearish():
        return 12
    if is_doji_star_bullish():
        return 13
    if is_doji_star_bearish():
        return 14
    if is_meeting_lines_bullish():
        return 19
    if is_meeting_lines_bearish():
        return 20
    if is_belt_hold_bullish():
        return 21
    if is_belt_hold_bearish():
        return 22
    if is_side_by_side_bullish():
        return 23
    if is_side_by_side_bearish():
        return 0

    # Если никакой паттерн не найден — можем вернуть что-то вроде "0" или "15..18" (не реализованные)
    # но для полноты возвращаем условные 15..18 (или 0). Ниже вариант с 15.
    return 15

@jit(nopython=True)
def all_True_any_False(closes: np.ndarray, opens: np.ndarray, numval: int, variant: str, types: bool, count: int = None) -> bool:
    num = numval+1
    closes = closes[-num:-1]
    opens = opens[-num:-1]

    comparisons = closes < opens

    if count is not None:
        if types:
            return np.sum(comparisons) >= count
        elif not types:
            return np.sum(~comparisons) >= count

    if variant == 'any':
        if types:
            return np.any(comparisons)
        else:
            return np.any(~comparisons)
    elif variant == 'all':
        if types:
            return np.all(comparisons)
        else:
            return np.all(~comparisons)
        
def diff(open, close):
    return (close-open)/open


def get_atr(candels, period):
    closes = candels[:, 4]
    highs = candels[:, 2]
    lows = candels[:, 3]
    
    atr = talib.ATR(highs, lows, closes, period)
    return atr[-1]