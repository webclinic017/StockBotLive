import json
import math
import requests
from config import *
import random


#states = {0:"Hold", 1:"Buy", 2:"Sell"}
states = {0:"BUY", 1:"SELL"}

BASE_URL = "https://paper-api.alpaca.markets"
ACCOUNT_URL = f"{BASE_URL}/v2/account"
ORDERS_URL = f"{BASE_URL}/v2/orders"
HEADERS = {'APCA-API-KEY-ID': API_KEY, 'APCA-API-SECRET-KEY': SECRET_KEY}
MARKET_URL = "wss://stream.data.sandbox.alpaca.markets/v2/{source}"




def getTotalMoney(agent, close_values, stocks):
  sum = 0
  for s in stocks:
    sum += agent.equity[s] + agent.inventory[s]*close_values[s]
  return sum

def getTotalCash(agent, stocks):
    sum=0
    for s in stocks:
        sum += agent.equity[s]
    return sum

def get_cash():
    r = get_account()
    return float(r['cash'])

def get_equity():
    r = get_account()
    return float(r['equity'])

def get_account():
    r = requests.get(ACCOUNT_URL, headers=HEADERS)
    return json.loads(r.content)


def create_order(symbol, quantity, side, type, time_in_force):
    data = {
        "symbol": symbol,
        "qty": quantity,
        "side": side,
        "type": type,
        "time_in_force": time_in_force
    }

    r = requests.post(ORDERS_URL, json=data, headers=HEADERS)

    return json.loads(r.content)


def get_orders():
    r = requests.get(ORDERS_URL, headers=HEADERS)

    return json.loads(r.content)

def bot_order(action, stock, close, agent, profit_data):

    equity = agent.equity[stock]
    inventory = agent.inventory[stock]

    buy = math.floor(equity / close)
    sell = inventory

    #print(f"*{action}*")
    if (action == 1 and inventory == 0) or (action == 0 and equity - (buy * close) <= 0) or (
            action == 0 and buy <= 0):
        print("Hold due to circumstances {}".format(action))
    elif action == 0 and equity - (buy * close) >= 0:  # buy
        #print(f'Stock : {stock} Equity : {equity} Inventory : {inventory} Close : {close}')
        profit_data[stock][0] = agent.equity[stock]
        agent.equity[stock] -= buy * close
        agent.inventory[stock] += buy
        #sell_option = 1
        create_order(stock, buy, "buy", "market", "gtc")

    elif action == 1 and inventory > 0:  # sell
        agent.equity[stock] += sell * close
        agent.inventory[stock] = 0
        # sell_option = 0
        create_order(stock, sell, "sell", "market", "gtc")
        profit_data[stock][1] = agent.equity[stock]

    print(f"{stock} : {states[action]}")
