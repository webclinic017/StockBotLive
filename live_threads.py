from queue import Queue
from threading import Thread
from stream import *
from time import *
from handling_live import *
from datetime import *
import numpy as np

q = database

datacenter = dict()

for s in stocks:
    datacenter.update({s : []})

print(datacenter)

# A thread that produces data
def stream(out_q):
    print("stream thread connected")
    stream_live()


# A thread that consumes data
def bot(in_q):
    print("bot thread connected")
    while True:

        if not in_q.empty():
            try:

                while not in_q.empty():
                    data = in_q.get()
                    stock_name = data[0]
                    stock_data_live_length = datacenter[stock_name]
                    input_data = [data[2], data[3], data[4], data[5], data[6]]
                    datacenter[stock_name].append(input_data)

                    if len(stock_data_live_length) > MAX_DATA_LENGTH:
                        datacenter[stock_name] = datacenter[stock_name][-MAX_DATA_LENGTH:]

                    historical_data = getHistoricalData(stock_name, MAX_DATA_LENGTH - stock_data_live_length)

                    live_data = np.array(datacenter[stock_name])
                    print(historical_data.shape)
                    print(live_data.shape)






                '''
                make sure to iterate through the whole queue until it is empty
                
                get name tag on the data and set a key value to it (used for the dict)
                access the datacenter dict and appened new data in proper format (c, o, h, l, v)
                to the datacenter
                
                pass data after updating to handling_live methods
                this will convert data into sizable image chunk which will be fed
                into the model
                
                after the model yeilds an action the proper action will be taken and 
                the alpaca will udpate on that stock promptly 
                '''
            except:
                print('failed to run')

        # Process the data
        sleep(60)



t1 = Thread(target=stream, args=(q,))
t2 = Thread(target=bot, args=(q,))
t1.start()
t2.start()
