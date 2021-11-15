import json
import datetime
import requests

response = requests.get("https://api.polygon.io/v2/aggs/ticker/"
                        "PLUG/range/1/minute/2021-11-10/2021-11-12"
                        "?adjusted=true&sort=desc&"
                        "limit=200"
                        "&apiKey=Qa633FUQ58r2IlTO5byH2xqSIOKy7vHh")


for i in range(1, 200):
    ms = response.json()['results'][i]['t']

    date = datetime.datetime.fromtimestamp(ms / 1000.0)
    date = date.strftime('%Y-%m-%d %H:%M:%S')
    print(date)

