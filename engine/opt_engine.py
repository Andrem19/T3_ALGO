import shared_vars as sv

def execute(settings, i, data):
    signal = settings['signal']
    tp_perc = settings['tp']
    sl_perc = settings['sl']
    
    open_price = data[i][1]
    open_time = data[i][0]
    stop_loss = open_price * (1+sl_perc) if signal == 2 else open_price * (1-sl_perc)
    take_profit = open_price * (1+tp_perc) if signal == 1 else open_price * (1-tp_perc)
    
    fee = settings['qty'] * open_price * 0.00045 * 2
    time_limit = settings['hours_to_exp'] * 60
    it = i-1

    amount = settings['qty']
    opt_amount = settings['optq']
    
    
    type_of_close = ''
    profit = 0
    close_price = 0
    
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
            close_price=stop_loss
            break
        
        if signal == 2 and high > stop_loss:
            type_of_close = 'stop_loss'
            profit = (open_price-stop_loss)*amount
            close_price=stop_loss
            break    
        
        #take_profit
        if signal == 1 and high > take_profit:
            type_of_close = 'take_profit'
            profit = (take_profit-open_price)*amount
            close_price = take_profit
            break
        
        if signal == 2 and low < take_profit:
            type_of_close = 'take_profit'
            profit = (open_price-take_profit)*amount
            close_price = take_profit
            break
        
    
    opt_profit = 0
    if signal == 1:
        if sv.opt['strike'] < close_price:
            opt_profit = -(sv.opt['askPrice']*opt_amount)
        elif close_price < sv.opt['strike']:
            opt_profit = ((sv.opt['strike']-close_price)*opt_amount)-(sv.opt['askPrice']*opt_amount)
        
    # print('OPT: ', sv.opt['strike'], open_price, close_price, opt_amount, sv.opt['askPrice'])
    
    pr = profit - fee
    report = {
        'open_time': open_time,
        'close_time': data[it][0],
        'type_of_signal': signal,
        'type_of_close': type_of_close,
        'profit': pr+opt_profit,
        'duration': duration,
        'fut_profit': profit - fee,
        'opt_profit': opt_profit,
        'cost_metric': sv.cost_metric,
    }
    return report, it