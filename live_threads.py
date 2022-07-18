from threading import Thread
from stream import *
from time import *
from handling_live import *
from datetime import *
import numpy as np
import trade
from variables import *
from static import *
from pandas import *
import stockstats

model_name = 'C:/Users/nirmi/PycharmProjects/StockBotLive/model_ema_sma_1.h5py'
#model_name = '/Users/nirmi/PycharmProjects/StockBotLive/model_ema_sma_1.h5py'
#model_name = 'C:/Users/nir/PycharmProjects/StockBotLive/model/stockbot/model_3_4_20'
agent = Agent(stocks=stocks, TIME_RANGE=TIME_RANGE, PRICE_RANGE=PRICE_RANGE, is_eval=True, model_name=model_name)

data = {'close': [], 'high': [], 'low': [], 'open': [], 'volume': []}
for s in stocks:
    datacenter.update({s: pd.DataFrame(data)})
    close_values.update({s: 0})
    fb.update({s: 1 / (len(stocks))})


trade_equities(agent, fb, TOTAL_EQUITY, close_values, 0)

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

                input_data = {'close': data[2], 'high': data[3], 'low': data[4], 'open': data[5],
                              'volume': data[6]}
                datacenter[stock_name] = datacenter[stock_name].append(input_data, ignore_index=True)
                stock_data_live_length = len(datacenter[stock_name]['close'])
                live_data = None
                historical_data = None
                if stock_data_live_length > MAX_DATA_LENGTH:
                    live_data = datacenter[stock_name][-MAX_DATA_LENGTH:]
                else:
                    live_data = datacenter[stock_name]

                historical_data = datacenter[stock_name]
                close_values[stock_name] = datacenter[stock_name]['close'].values[-1]
                if stock_data_live_length >= MAX_DATA_LENGTH:
                    processed_live_data = getStockDataLive(stock_name, historical_data, live_data)


                    live_state = getStateLive(data = processed_live_data)
                    #Stock Bot acting for Stock

                    action = agent.act(live_state)

                    trade.bot_order(action=action,
                                    stock=stock_name,
                                    close=datacenter[stock_name]['close'].values[-1],
                                    agent=agent,
                                    profit_data=profit_data)

                    #Forecast bot

                    forecast = 0

                    #Get Peformance
                    performance = getBotPeformance(profit_data=profit_data, stock=stock_name)

                    #Update FB scores
                    update_fb(fb_scores=fb,
                              performance=performance,
                              forecast=forecast,
                              stock=stock_name,
                              stocks=stocks)

                    #Trading

                    trade_equities(agent=agent,
                                   fb_values=fb,
                                   total_money=trade.get_equity(agent, close_values, stocks),
                                   close_values=close_values,
                                   init_cash=0)
                    total_equity = 0
                    for s in stocks:
                        total_equity += agent.equity[s] + agent.inventory[s] * close_values[s]
                    print(total_equity)




t1 = Thread(target=stream, args=(database,))
t2 = Thread(target=bot, args=(database,))
t2.start()
t1.start()

