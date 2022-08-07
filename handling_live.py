import matplotlib.pyplot as plt

import trade

plt.style.use('fivethirtyeight')

import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as pdr
import stockstats
import cv2
from PIL import Image
import math
from variables import *
import random

from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, RobustScaler, MaxAbsScaler, PowerTransformer
from keras.applications.xception import Xception
import tensorflow as tf
import math


def scale_list(l, to_min, to_max):
    def scale_number(unscaled, to_min, to_max, from_min, from_max):
        return (to_max - to_min) * (unscaled - from_min) / (from_max - from_min) + to_min

    if len(set(l)) == 1:
        return [np.floor((to_max + to_min) / 2)] * len(l)
    else:
        return [scale_number(i, to_min, to_max, min(l), max(l)) for i in l]


def getStateLive(data):
    closing_values = data[0]
    cnt = len(closing_values)
    stock_ema3 = data[1]
    stock_sma3 = data[2]
    close_minus_open = data[3]

    graph_ema3 = list(np.round(scale_list(stock_ema3[cnt - TIME_RANGE:cnt], 0, half_scale_size - 1), 0))
    graph_sma3 = list(np.round(scale_list(stock_sma3[cnt - TIME_RANGE:cnt], 0, half_scale_size - 1), 0))
    graph_close_minus_open = list(
        np.round(scale_list(close_minus_open[cnt - TIME_RANGE:cnt], 0, half_scale_size - 1), 0))

    blank_matrix_close = np.zeros(shape=(half_scale_size, TIME_RANGE))
    x_ind = 0
    for ma, c in zip(graph_ema3, graph_sma3):
        if np.isnan(ma):
            ma = 0
        if np.isnan(c):
            c = 0
        blank_matrix_close[int(ma), x_ind] = 1
        blank_matrix_close[int(c), x_ind] = 2
        x_ind += 1

    blank_matrix_close = blank_matrix_close[::-1]

    blank_matrix_diff = np.zeros(shape=(half_scale_size, TIME_RANGE))
    x_ind = 0
    for v in graph_close_minus_open:
        if np.isnan(v):
            v = 0
        blank_matrix_diff[int(v), x_ind] = 3
        x_ind += 1
        # flip x scale so high number is atop, low number at bottom - cosmetic, humans only
    blank_matrix_diff = blank_matrix_diff[::-1]

    blank_matrix = np.vstack([blank_matrix_close, blank_matrix_diff])

    if 1 == 2:
        # graphed on matrix
        plt.imshow(blank_matrix)
        plt.show()

    x_train = [blank_matrix]
    x_train = np.array(x_train)
    x_train = x_train.reshape(x_train.shape[0], TIME_RANGE, PRICE_RANGE, 1)
    x_train = x_train.astype('float32')

    return x_train


'''
def getStateLive(data, sell_option, TIME_RANGE, PRICE_RANGE):
    closing_values = data[0]
    t = len(closing_values)  # Finale value, repersenting live value
    macd = data[1]
    macds = data[2]
    # print(closing_values)
    half_scale_size = int(PRICE_RANGE / 2)

    graph_closing_values = list(np.round(scale_list(closing_values[t - TIME_RANGE:t], 0, half_scale_size - 1), 0))
    macd_data_together = list(
        np.round(scale_list(list(macd[t - TIME_RANGE:t]) + list(macds[t - TIME_RANGE:t]), 0, half_scale_size - 1), 0))
    graph_macd = macd_data_together[0:PRICE_RANGE]
    graph_macds = macd_data_together[PRICE_RANGE:]

    blank_matrix_macd = np.zeros((half_scale_size, TIME_RANGE, 3), dtype=np.uint8)
    x_ind = 0
    for s, d in zip(graph_macds, graph_macd):
        if math.isnan(s):
            s = PRICE_RANGE - 10
        if math.isnan(d):
            d = PRICE_RANGE - 10
        blank_matrix_macd[int(s), x_ind] = (0, 0, 255)
        blank_matrix_macd[int(d), x_ind] = (255, 175, 0)
        x_ind += 1
    blank_matrix_macd = blank_matrix_macd[::-1]

    blank_matrix_close = np.zeros((half_scale_size, TIME_RANGE, 3), dtype=np.uint8)
    x_ind = 0
    if sell_option == 1:
        close_color = (0, 255, 0)  # GREEN
    else:
        close_color = (255, 0, 0)  # RED

    for v in graph_closing_values:
        blank_matrix_close[int(v), x_ind] = close_color
        x_ind += 1
    blank_matrix_close = blank_matrix_close[::-1]

    blank_matrix = np.vstack([blank_matrix_close, blank_matrix_macd])

    if 1 == 2:
        # graphed on matrix
        plt.imshow(blank_matrix)
        plt.show()
        # print('worked')

    return [blank_matrix]
'''


def getHistoricalData(key, length_data):
    return 0


def getStockDataLive(key, historical_data, live_data):
    stock_data = None
    stats = None
    if len(live_data) < MAX_DATA_LENGTH:
        stock_data = historical_data[-(MAX_DATA_LENGTH - len(live_data)):] + live_data
    else:
        stock_data = live_data

    ema3 = stock_data['close'].ewm(span=3, adjust=False).mean()
    ema3 = ema3.fillna(method='bfill')
    ema3 = list(ema3.values)

    sma3 = stock_data['open'].rolling(3).mean()
    sma3 = sma3.fillna(method='bfill')
    sma3 = list(sma3.values)

    stock_opens = stock_data['open'].rolling(3).mean()
    stock_opens = stock_opens.fillna(method='bfill')
    stock_opens = list(stock_opens.values)

    closing_values = list(np.array(stock_data['close']))

    stock_closes = stock_data['close'].ewm(span=3, adjust=False).mean()
    stock_closes = stock_closes.fillna(method='bfill')
    stock_closes = list(stock_closes.values)

    closing_values = list(np.array(stock_data['close']))

    close_minus_open = list(np.array(stock_closes) - np.array(stock_opens))

    return_data = [closing_values, ema3, sma3, close_minus_open]

    return return_data


# prints formatted price
def formatPrice(n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))


def fix_input(state):
    state = np.array(state)
    img_rows, img_cols = TIME_RANGE, PRICE_RANGE
    state = np.reshape(state, (state.shape[0], img_rows, img_cols, 3))
    state = state.astype('float32')
    return state


import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, LSTM, Dropout
import tensorflow as tf

import numpy as np
import random
from collections import deque


class Agent:
    def __init__(self, stocks, PRICE_RANGE, TIME_RANGE, is_eval=False, model_name=""):
        self.price_range = PRICE_RANGE
        self.time_range = TIME_RANGE

        self.action_size = 2  # sit, buy, sell
        self.memory = deque(maxlen=500_000)
        self.model_name = model_name
        self.is_eval = is_eval
        self.inventory = dict()
        self.equity = dict()
        self.stocks = stocks

        for s in stocks:
            self.inventory.update({s: 0})
            self.equity.update({s: 0})

        self.gamma = 0.92
        self.epsilon = 0.8
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.9988

        self.model = load_model(model_name)
        print('Loaded Model ', model_name)
        '''
    def create_model(self):
        
        input_shape_1 = (self.time_range, self.price_range, 3)

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape_1))
        model.add(Conv2D(64, kernel_size=(4, 4), activation='relu'))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(learning_rate=.001), metrics=['accuracy'])
        return 0
        '''

    def act(self, state):
        # action = self.model.predict(state)

        return 0  # np.argmax(action)

    def expReplay(self, batch_size):
        mini_batch = []
        l = len(self.memory)
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])
        #	mini_batch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in mini_batch:
            target = reward
            state = fix_input(state)
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(fix_input(next_state))[0])

            target_f = self.model.predict(state)
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def update_fb(fb_scores, stocks, forecast, stock):
    '''
        get bot peformance, get forecast
        make sure fb scores add up to 1
    '''
    performances[stock] += forecast
    normalize_data(stocks)

    print(fb_scores)


def getBotPeformance(profit_data):
    # DATA should be
    # Equity of bot right before it buys | Equity of bot right after it sells
    for s in stocks:
        p_d = profit_data[s]
        performance = p_d[1] / p_d[0]
        performances[s] = performance


def normalize_data(stocks):
    sum_2 = 0
    temp = []
    for s in stocks:
        sum_2 += performances[s]
    for s in stocks:
        score = performances[s]
        fb[s] = (score / sum_2)


def trade_equities(agent, fb_values, total_money, close_values, init_cash):
    print("Trading Equities")
    if init_cash > 0:
        pool = init_cash
    else:
        pool = 0
    # First Pass
    for s in agent.stocks:
        agent_fb = fb_values[s]
        agent_inventory = agent.inventory[s]
        close = close_values[s]
        cash = agent.equity[s]
        live_money = agent_inventory * close
        agent_initial_equity = cash + live_money
        equity = total_money * agent_fb
        change_equity = equity - agent_initial_equity
        print(
            f'Stock : {s} Equity : {agent_initial_equity} Inventory : {agent.inventory[s]} Close : {close_values[s]} Deserved Equity : {equity}')

        if change_equity < 0:
            if change_equity + cash >= 0:
                agent.equity[s] -= abs(change_equity)
                pool += abs(change_equity)
            else:
                print(s, ' SOLD')
                change_equity += cash
                pool += cash
                sell_2 = 0
                if agent_inventory * close >= change_equity and agent_inventory >= 0:
                    sell_2 = math.floor(abs(change_equity) / close)
                elif agent_inventory >= 0:
                    sell_2 = agent_inventory

                pool += sell_2*close

                trade.create_order(s, sell_2, "sell", "market", "gtc")
                agent.inventory[s] -= sell_2


            print('pool1', pool)
    #second pass
    for s in agent.stocks:
        agent_fb = fb_values[s]
        agent_inventory = agent.inventory[s]
        close = close_values[s]

        cash = agent.equity[s]
        live_money = agent_inventory * close
        agent_initial_equity = cash + live_money
        equity = total_money * agent_fb
        change_equity = equity - agent_initial_equity

        if change_equity >= 0:
            if pool >= change_equity:
                agent.equity[s] += change_equity
                pool -= abs(change_equity)
            else:
                agent.equity[s] += pool
                pool = 0
    print('pool', pool)
    # Final Third Pass
    # print('***', pool)
    if pool > 0:
        for s in stocks:
            agent.equity[s] += pool / (len(stocks))
