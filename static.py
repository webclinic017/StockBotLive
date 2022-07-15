import time
import datetime
import pandas as pd
import numpy as np
import pandas_datareader as pdr
from variables import *


def getStockData(stock):
    # print(stock)
    # stock_data = pd.read_csv(f'C:/Users/nirmi/PycharmProjects/StockBotLive/Data/{stock}.txt', parse_dates=True, index_col='Date')
    stock_data = pdr.get_data_tiingo(stock, api_key='9d4f4dacda5024f00eb8056b19009f32e58b38e5')[-50:]
    close = list(np.array(stock_data['close']))
    open = list(np.array(stock_data['open']))
    high = list(np.array(stock_data['high']))
    low = list(np.array(stock_data['low']))
    volume = list(np.array(stock_data['volume']))

    # add Moving Averages to all lists and back fill resulting first NAs to last known value

    return [close, high, low, open, volume]


def file_data(STOCKS):
    datatemp = []
    datac = []
    for s in STOCKS:
        datatemp.append(getStockData(s))
    p = 0
    for i in range(len(datatemp[0][0])):
        for j in range(len(STOCKS) - 1):
            data = datatemp[j]
            p += 1
            datac.append([STOCKS[j], p, data[0][i], data[1][i], data[2][i], data[3][i], data[4][i]])
    return datac


def stream_static(datac):
    for d in datac:
        database.put([d[0], d[1], d[2], d[3], d[4], d[5], d[6]])
        print(f"Stock : {d[0]}, {d[1]}, {d[2]}")
