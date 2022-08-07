from threading import Thread
from streamv2 import *
from time import *
from handling_live import *
from datetime import *
import numpy as np
import trade
from variables import *
from static import *
from variables import *

model_name = 'C:/Users/nirmi/PycharmProjects/StockBotLive/model_ema_sma_1.h5py'
# model_name = '/Users/nirmi/PycharmProjects/StockBotLive/model_ema_sma_1.h5py'
# model_name = 'C:/Users/nir/PycharmProjects/StockBotLive/model/stockbot/model_3_4_20'
agent = Agent(stocks=stocks, TIME_RANGE=TIME_RANGE, PRICE_RANGE=PRICE_RANGE, is_eval=True, model_name=model_name)

data = {'close': [], 'high': [], 'low': [], 'open': [], 'volume': []}
for s in stocks:
    datacenter.update({s: pd.DataFrame(data)})
    close_values.update({s: 0})
    live_values.update({s: None})
    fb.update({s: 1 / (len(stocks))})
    performances.update({s: 1})

#trade_equities(agent, fb, TOTAL_EQUITY, close_values, 0)

for s in stocks:
    agent.equity[s] = TOTAL_EQUITY * fb[s]
print(agent.equity)
for s in stocks:
    # Initial Equity
    profit_data.update({s: [agent.equity[s], agent.equity[s]]})

'''
Get Static Data
'''
datac = file_data(stocks)

'''
Get Static Data
'''


# A thread that produces data
def stream(out_q):
    print("stream thread connected")
    #stream_live()
    stream_static(datac)


# A thread that consumes data
def bot(in_q):
    print("bot thread connected")

    while True:
        if not in_q.empty():
            while not in_q.empty():
                data = in_q.get()
                stock_name = data[0]

                t = data[1]
                #print('BRUH', stock_name, t)

                input_data = {'close': data[2], 'high': data[3], 'low': data[4], 'open': data[5],
                              'volume': data[6]}
                datacenter[stock_name] = datacenter[stock_name].append(input_data, ignore_index=True)
                stock_data_live_length = len(datacenter[stock_name]['close'])
                print('Put : ', stock_name, stock_data_live_length, input_data)
                live_data = None
                historical_data = None
                if stock_data_live_length > MAX_DATA_LENGTH:
                    live_data = datacenter[stock_name][-MAX_DATA_LENGTH:]
                else:
                    live_data = datacenter[stock_name]

                historical_data = datacenter[stock_name]
                close_values[stock_name] = datacenter[stock_name]['close'].values[-1]
                time_stamps[stock_name] = t
                print('closing_values ', close_values)
                if stock_data_live_length >= MAX_DATA_LENGTH:
                    processed_live_data = getStockDataLive(stock_name, historical_data, live_data)

                    live_state = getStateLive(data=processed_live_data)
                    # Stock Bot acting for Stock

                    action = agent.act(live_state)

                    trade.bot_order(action=0,
                                    stock=stock_name,
                                    close=datacenter[stock_name]['close'].values[-1],
                                    agent=agent,
                                    profit_data=profit_data)
                    # Get Peformance
                    performances = getBotPeformance(profit_data=profit_data)

                    forecast = 0

                    if action == 1:
                        forecast = .1
                    else:
                        forecast = -.1


                    # Update FB scores
                    update_fb(fb_scores=fb,
                              stocks=stocks,
                              forecast=0,
                              stock=stock_name)

                    # Trading
                    length = len(datacenter[stocks[0]]['close'])
                    if not any(length != len(datacenter[s]['close']) for s in stocks):
                        total_equity = 0
                        #total_equity = trade.get_equity()
                        for s in stocks:
                            total_equity += agent.equity[s] + agent.inventory[s] * close_values[s]
                        print('total equity ', total_equity)
                        trade_equities(agent=agent,
                                       fb_values=fb,
                                       total_money=total_equity,
                                       close_values=close_values,
                                       init_cash=0)

t1 = Thread(target=stream, args=(database,))
t2 = Thread(target=bot, args=(database,))
t2.start()
t1.start()
t1.join()
t2.join()
