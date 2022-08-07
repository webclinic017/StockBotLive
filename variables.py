from queue import Queue
import pip
#pip.main(['install','stockstats == 0.4.1'])
#pip.main(['install','tensorflow'])
#pip.main(['install','sklearn'])
#pip.main(['install', 'alpaca_trade_api'])
#pip.main(['install', 'ibapi'])

TOTAL_EQUITY = 100_000
equity_center = dict()
datacenter = dict()
fb = dict()
close_values = dict()
time_stamps = dict()
performances = dict()
live_values = dict()
#Profit data : stock - [total_profit, initial_equity]
profit_data = dict()

TIME_RANGE, PRICE_RANGE = 20, 20
half_scale_size = int(PRICE_RANGE/2)
MAX_DATA_LENGTH = 20
if MAX_DATA_LENGTH < TIME_RANGE:
    MAX_DATA_LENGTH = TIME_RANGE

#stocks = ['AFL', 'DOX', 'HON', 'BCE', 'BRK.B', 'L', 'PEP', 'KO', 'LIT']
stocks = ['AAPL', 'TSLA', 'AMZN', 'NVDA']#,'GE','GS','HD','IBM','INTC','JNJ','JPM','KO','MCD','MMM','MRK','MSFT']#,'NKE','PFE','PG','TRV','UNH','UTX','V','VZ','WMT','XOM']

database = Queue()


am_stocks = []
for i in range(len(stocks)):
    am_stocks.append(f"AM.{stocks[i]}")



