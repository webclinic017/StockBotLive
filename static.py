import threading
import time
import datetime
import pandas as pd
import numpy as np
import pandas_datareader as pdr
from variables import *
import alpaca_trade_api as tradeapi
from config import *
from threading import *

api_key = API_KEY
api_secret = SECRET_KEY
base_url = 'https://paper-api.alpaca.markets'
api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')


def getStockData(stock, time_frame_):
    # print(stock)
    #stock_data = pd.read_csv(f'C:/Users/nirmi/PycharmProjects/StockBotLive/Data/{stock}.txt', parse_dates=True, index_col='Date')
    #stock_data = pdr.get_data_tiingo(stock, api_key='9d4f4dacda5024f00eb8056b19009f32e58b38e5')[-100:]

    stock_data = api.get_bars(symbol=stock, timeframe=time_frame_, limit=100, start='2022-08-01').df
    close = list(np.array(stock_data['close']))
    open = list(np.array(stock_data['open']))
    high = list(np.array(stock_data['high']))
    low = list(np.array(stock_data['low']))
    volume = list(np.array(stock_data['volume']))
    #time_stamp = list(np.array(stock_data.index.astype(np.int64)))
    print(stock, close)

    # add Moving Averages to all lists and back fill resulting first NAs to last known value

    return [close, high, low, open, volume]

#
def file_data(STOCKS):
    datatemp = []
    datac = []
    for s in STOCKS:
        datatemp.append(getStockData(s, '1Min'))
    p = 0
    for i in range(len(datatemp[0][0])):
        for j in range(len(STOCKS)):
            data = datatemp[j]
            datac.append([STOCKS[j], p, data[0][i], data[1][i], data[2][i], data[3][i], data[4][i]])
            p += 1
    return datac


def stream_static(datac):
    for d in datac:
        database.put([d[0], d[1], d[2], d[3], d[4], d[5], d[6]])
        #print(f"Stock : {d[0]}, {d[1]}, {d[2]}")

