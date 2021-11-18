from threading import Thread
from stream import *
from time import *
from handling_live import *
from datetime import *
import numpy as np
import pandas as pd
import trade
from variables import *

model_name = 'model_1'
agent = Agent(stocks=stocks, TIME_RANGE=TIME_RANGE, PRICE_RANGE=PRICE_RANGE, is_eval=True, model_name=model_name)

data = {'close': [], 'high': [], 'low': [], 'open': [], 'volume': []}
for s in stocks:
    datacenter.update({s: pd.DataFrame(data)})


# A thread that produces data
def stream(out_q):
    print("stream thread connected")
    stream_live()


# A thread that consumes data
def bot(in_q):
    print("bot thread connected")

    while True:
            try:
                if not in_q.empty():
                    while not in_q.empty():
                        data = in_q.get()
                        print(data)
                        stock_name = data[0]

                        input_data = {'close': data[2], 'high': data[3], 'low': data[4], 'open': data[5],
                                      'volume': data[6]}
                        datacenter[stock_name] = datacenter[stock_name].append(input_data, ignore_index=True)
                        stock_data_live_length = len(datacenter[stock_name]['close'])

                        if stock_data_live_length > MAX_DATA_LENGTH:
                            datacenter[stock_name] = datacenter[stock_name][-MAX_DATA_LENGTH:]

                        # historical_data = getHistoricalData(stock_name, MAX_DATA_LENGTH - stock_data_live_length)

                        live_data = datacenter[stock_name]
                        if stock_data_live_length >= MAX_DATA_LENGTH:
                            processed_live_data = getStockDataLive(stock_name, [], live_data)
                            live_state = getStateLive(data=processed_live_data,
                                                      sell_option=1,
                                                      TIME_RANGE=TIME_RANGE,
                                                      PRICE_RANGE=PRICE_RANGE)

                            action = agent.act(live_state)

                            trade.bot_order(action=action,
                                            stock=stocks,
                                            close=datacenter[stock_name]['close'].values[-1],
                                            inventory=agent.inventory[stock_name],
                                            equity=agent.equity[stock_name])
            except:
                print('failed to run')
            sleep(1)

t1 = Thread(target=stream, args=(database,))
t2 = Thread(target=bot, args=(database,))
t1.start()
t2.start()


for i in range(200):
    database.put(['TSLA', 1, 2, 3, 4, 5, 6])

