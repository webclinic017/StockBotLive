import json
import math
import requests
from config import *

#states = {0:"Hold", 1:"Buy", 2:"Sell"}
states = {0:"BUY", 1:"SELL"}

BASE_URL = "https://paper-api.alpaca.markets"
ACCOUNT_URL = f"{BASE_URL}/v2/account"
ORDERS_URL = f"{BASE_URL}/v2/orders"
HEADERS = {'APCA-API-KEY-ID': API_KEY, 'APCA-API-SECRET-KEY': SECRET_KEY}



def get_equity():
    r = get_account()
    return float(r['last_equity'])

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

def bot_order(action, stock, close, agent):

    equity = agent.equity[stock]
    inventory = agent.inventory[stock]

    buy = math.floor(equity / close)
    sell = inventory
    if (action == 1 and inventory == 0) or (action == 0 and equity - (buy * close) <= 0) or (
            action == 0 and buy <= 0):
        print("Hold due to circumstances {}".format(action))
    elif action == 0 and equity - (buy * close) >= 0:  # buy
        agent.equity[stock] -= buy * close
        agent.inventory[stock] += buy
        #sell_option = 1
        create_order(stock, buy, "buy", "market", "gtc")
    elif action == 1 and inventory > 0:  # sell
        agent.equity[stock] += sell * close
        agent.inventory[stock] -= sell
        # sell_option = 0
        create_order(stock, sell, "sell", "market", "gtc")

    print(f"{stock} : {states[action]}")
