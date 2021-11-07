import config
import websocket, json
from queue import Queue

database = Queue()

def on_open(ws):
    print("stream opened")
    auth_data = {
        "action": "authenticate",
        "data": {"key_id": config.API_KEY, "secret_key": config.SECRET_KEY}
    }

    ws.send(json.dumps(auth_data))

    listen_message = {"action": "listen", "data": {"streams": ["AM.TSLA"]}}

    ws.send(json.dumps(listen_message))


def on_message(ws, message):

    data = json.loads(message)['data']
    database.put(data['c'])
    print(database)

def on_close(ws):
    print("closed connection")


def stream_live():
    socket = "wss://data.alpaca.markets/stream"
    ws = websocket.WebSocketApp(socket, on_open=on_open, on_message=on_message, on_close=on_close)
    ws.run_forever()