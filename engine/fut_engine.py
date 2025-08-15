import shared_vars as sv
import settings as set

import copy

def execute(settings, i, data):
    signal=settings['signal']
    tp_perc = 0.02
    sl_perc = 0.02
    
    open_price = data[i][1]
    open_time = data[i][0]
    stop_loss = open_price * (1+sl_perc) if signal == 2 else open_price * (1-sl_perc)
    take_profit = open_price * (1+tp_perc) if signal == 1 else open_price * (1-tp_perc)
    
    fee = set.base_amount * open_price * 0.00045 * 2
    time_limit = 1430
    it = i-1

    amount = 1000 / open_price
    type_of_close = ''
    profit = 0
    
    while True:
        it+=1
        high = data[it][2]
        low = data[it][3]
        duration = ((data[it][0]+1)-open_time) // 60000
        
        
        #time-limit
        if duration >=time_limit:
            type_of_close = 'time_limit'
            close_price = data[it][1]
            if signal == 1:
                profit = (close_price-open_price)*amount
            else:
                profit = (open_price-close_price)*amount
            break

        #stop loss
        if signal == 1 and low < stop_loss:
            type_of_close = 'stop_loss'
            profit = (stop_loss-open_price)*amount
            break
        
        if signal == 2 and high > stop_loss:
            type_of_close = 'stop_loss'
            profit = (open_price-stop_loss)*amount
            break    
        
        #take_profit
        if signal == 1 and high > take_profit:
            type_of_close = 'take_profit'
            profit = (take_profit-open_price)*amount
            break
        
        if signal == 2 and low < take_profit:
            type_of_close = 'take_profit'
            profit = (open_price-take_profit)*amount
            break
        

             
    
    report = {
        'open_time': open_time,
        'close_time': data[it][0],
        'type_of_signal': signal,
        'type_of_close': type_of_close,
        'profit': profit - fee,
        'duration': duration,
        'metrics': copy.deepcopy(sv.metrics)
    }
    return report, it