from threading import Thread
from stream import *
from time import *
from handling_live import *
from datetime import *
import numpy as np
import pandas as pd
import trade
from variables import *

model_name = '/content/drive/MyDrive/StockBot/models/stock_bot_comp/CNN/model_3/model_3_4_20'
#model_name = 'C:/Users/nir/PycharmProjects/StockBotLive/model/stockbot/model_3_4_20'
agent = Agent(stocks=stocks, TIME_RANGE=TIME_RANGE, PRICE_RANGE=PRICE_RANGE, is_eval=True, model_name=model_name)

data = {'close': [], 'high': [], 'low': [], 'open': [], 'volume': []}
for s in stocks:
    datacenter.update({s: pd.DataFrame(data)})
    close_values.update({s: 0})
    fb.update({s: 1 / (len(stocks))})

trade_equities(agent, fb, trade.get_equity(), close_values, trade.get_cash())

for s in stocks:
    print(agent.equity[s])
    # Initial Equity
    profit_data.update({s: [agent.equity[s], agent.equity[s]]})



# A thread that produces data
def stream(out_q):
    print("stream thread connected")
    stream_live()


# A thread that consumes data
def bot(in_q):
    print("bot thread connected")

    while True:
        if not in_q.empty():
            while not in_q.empty():
                data = in_q.get()
                stock_name = data[0]

                input_data = {'close': data[2], 'high': data[3], 'low': data[4], 'open': data[5],
                              'volume': data[6]}
                datacenter[stock_name] = datacenter[stock_name].append(input_data, ignore_index=True)
                stock_data_live_length = len(datacenter[stock_name]['close'])

                if stock_data_live_length > MAX_DATA_LENGTH:
                    datacenter[stock_name] = datacenter[stock_name][-MAX_DATA_LENGTH:]

                # historical_data = getHistoricalData(stock_name, MAX_DATA_LENGTH - stock_data_live_length)

                live_data = datacenter[stock_name]
                close_values[stock_name] = datacenter[stock_name]['close'].values[-1]
                if stock_data_live_length >= MAX_DATA_LENGTH:
                    processed_live_data = getStockDataLive(stock_name, [], live_data)
                    live_state = getStateLive(data=processed_live_data,
                                              sell_option=1,
                                              TIME_RANGE=TIME_RANGE,
                                              PRICE_RANGE=PRICE_RANGE)


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
                              stock=stock_name)

                    #Trading

                    trade_equities(agent=agent,
                                   fb_values=fb,
                                   total_money=trade.get_equity(),
                                   close_values=close_values,
                                   cash=trade.get_cash())

        sleep(1)


t1 = Thread(target=stream, args=(database,))
t2 = Thread(target=bot, args=(database,))
t1.start()
t2.start()
