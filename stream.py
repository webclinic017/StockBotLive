import config
import websocket, json
from queue import Queue
import datetime

database = Queue()

stocks = ['TSLA', 'PLUG']
for i in range(len(stocks)):
    am_stocks = f'AM.{stocks[i]}'

def on_open(ws):
    print("stream opened")
    auth_data = {
        "action": "authenticate",
        "data": {"key_id": config.API_KEY, "secret_key": config.SECRET_KEY}
    }

    ws.send(json.dumps(auth_data))

    listen_message = {"action": "listen", "data": {"streams": am_stocks}}

    ws.send(json.dumps(listen_message))

def on_message(ws, message):

    data = json.loads(message)['data']

    ms = data['e']
    date = datetime.datetime.fromtimestamp(ms/1000.0)
    date = date.strftime('%Y-%m-%d %H:%M:%S')

    database.put([data['c']])

    print(database)
    print(date)

def on_close(ws):
    print("closed connection")


def stream_live():
    socket = "wss://data.alpaca.markets/stream"
    ws = websocket.WebSocketApp(socket, on_open=on_open, on_message=on_message, on_close=on_close)
    ws.run_forever()