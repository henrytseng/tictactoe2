from os import environ
from multiprocessing import Process, Queue
import time
import random

class SavedState(object):

    def __init__(self):
        self.q = Queue()
        self.p = Process()
        self.temp_path = environ['DATA_FOLDER'] if 'DATA_FOLDER' in environ else './data'
        self.filename = "{}.data".format(hex(random.getrandbits(16)))
        self.ts = time.time()
        self.states = []
        
    def reset(self):
        self.states = []
        self.ts = time.time()

    def push(self, board=None, player=None, move=None):
        state = (
            player,
            board.serialize(),
            move,
        )
        self.states.append(state)

    def store(self, winner=None):
        with open("{tmp}/{filename}".format(tmp=self.temp_path, filename=self.filename), 'a') as f:
            for i in self.states:
                payload = (
                    self.ts,
                    winner,
                ) + i
                print(payload)
                f.write(",".join(map(str, payload))+"\n")
