from queue import Queue
from threading import Thread
from stream import *
from time import *
from datetime import *

q = database

# A thread that produces data
def stream(out_q):
    print("stream thread connected")
    stream_live()


# A thread that consumes data
def bot(in_q):
    print("bot thread connected")
    while True:
        # Get some data
        data = in_q.get()
        # Process the data
        print("bot ping")
        sleep(30)


t1 = Thread(target=stream, args=(q,))
t2 = Thread(target=bot, args=(q,))
t1.start()
t2.start()