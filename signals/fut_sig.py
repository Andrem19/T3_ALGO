from calendar import weekday
from datetime import datetime
import shared_vars as sv
import helpers.tools_fut as tl_fut
import talib

import sys
import talib


def signal(i):
    dt = datetime.fromtimestamp(sv.data_fut[i][0]/1000)
    
    hour = dt.hour
    minute = dt.minute
    close_6h_ago = sv.data_fut[i-420][4]
    close_30h_ago = sv.data_fut[i-1800][4]
    close_now = sv.data_fut[i-1][4]

    signal = 0
    h_to_exp = 0
    opt_q = 0
    fut_qty, tp, sl = None, None, None
    
    # cand1h_1 = [0, sv.data_fut[i-60][1], max(sv.data_fut[i-60:i, 2]), min(sv.data_fut[i-60:i, 3]), sv.data_fut[i-1][4]]
    # cand1h_2 = [0, sv.data_fut[i-120][1], max(sv.data_fut[i-120:i-60, 2]), min(sv.data_fut[i-120:i-60, 3]), sv.data_fut[i-60][4]]
        
    # opens1h, highs1h, lows1h, closes1h = tl_fut.convert_timeframe(sv.data_fut[i-1220:i], 60, 20)

    sv.metrics['updown'] = 1

    signal = 1

        
    settings = {
        'signal': signal,
        'qty': fut_qty,
        'optq': opt_q,
        'tp': tp,
        'sl': sl,
        'hours_to_exp': h_to_exp,
    }
    return settings

