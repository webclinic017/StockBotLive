from queue import Queue
from threading import Thread
from stream import *
from time import *
from datetime import *

q = database

datacenter = dict()

for s in stocks:
    datacenter.update({s : None})

print(datacenter)

# A thread that produces data
def stream(out_q):
    print("stream thread connected")
    stream_live()


# A thread that consumes data
def bot(in_q):
    print("bot thread connected")
    while True:
        # Get some data
        print("bot ping")
        if not in_q.empty():
            try:
                data = in_q.get()
                '''
                get name tag on the data and set a key value to it (used for the dict)
                access the datacenter dict and appened new data in proper format (c, o, h, l, v)
                to the datacenter
                
                pass data after updating to handling_live methods
                this will convert data into sizable image chunk which will be fed
                into the model
                
                after the model yeilds an action the proper action will be taken and 
                the alpaca will udpate on that stock promptly 
                
                make sure to iterate through the whole queue until it is empty
                '''
            except:
                print('failed to run')

        # Process the data
        sleep(60)


'''
t1 = Thread(target=stream, args=(q,))
t2 = Thread(target=bot, args=(q,))
t1.start()
t2.start()
'''