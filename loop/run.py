import shared_vars as sv
import signals.fut_sig as fut_sg
import signals.opt_sig as opt_sg
import settings as set
import engine.fut_engine as feng_1
import engine.opt_engine as oeng_1
from datetime import datetime

async def run_loop():
    data_len = len(sv.data_fut)-2000
    i = 3000
    while i < data_len:
        settings = {'signal': 0}
        if set.with_opt:
            opt_sg.signal(i)
        else:
            settings = fut_sg.signal(i)
        
        if settings['signal'] == 0:
            i+=1
            continue
        
        report, it = None, None
        if set.with_opt:
            report, it = oeng_1.execute(settings, i, sv.data_fut)
        else:
            report, it = feng_1.execute(settings, i, sv.data_fut)
        
        if report is not None:
            sv.total+= report['profit']
            print(sv.total, datetime.fromtimestamp(report['open_time']/1000), report)
            print('\n\n')
            
            sv.positions_list.append(report)
            i=it+1