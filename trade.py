import json

import requests
from config import *

BASE_URL = "https://paper-api.alpaca.markets"
ACCOUNT_URL = f"{BASE_URL}/v2/account"
ORDERS_URL = f"{BASE_URL}/v2/orders"
HEADERS = {'APCA-API-KEY-ID': API_KEY, 'APCA-API-SECRET-KEY': SECRET_KEY}


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

create_order("PLUG", 990, "sell", "market", "gtc")
