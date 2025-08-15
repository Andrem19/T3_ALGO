from datetime import datetime
import shared_vars as sv
import helpers.tools_opt as tools_opt



def signal(i):
    dt = datetime.fromtimestamp(sv.data_fut[i][0]/1000)
    hour = dt.hour
    minute = dt.minute
    
    signal = 0
    h_to_exp = 0
    opt_q = 0
    fut_qty, tp, sl = None, None, None
    
    snap = sv.loader.get_snapshot_filtered(sv.data_fut[i-1][0], min_hours_to_expiry=3, max_hours_to_expiry=26, opt_type=None, max_moneyness_pct=0.01)
    
    opt_P = tools_opt.get_p_otm(snap, 'P', ['OTM'])
    
    
    settings = {
        'signal': signal,
        'qty': fut_qty,
        'optq': opt_q,
        'tp': tp,
        'sl': sl,
        'hours_to_exp': h_to_exp,
    }
    return settings