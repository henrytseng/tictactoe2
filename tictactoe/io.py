from os import environ
import time
import random

class SavedState(object):

    def __init__(self):
        self.temp_path = environ['TMP_FOLDER'] if 'TMP_FOLDER' in environ else './tmp'
        self.filename = "d.data".format(hex(random.getrandbits(16)))
        self.ts = time.time()
        self.states = []
        
    def retrieve(self):
        with open("{tmp}/game_states.data".format(tmp=self.temp_path), 'a+') as f:
            lines = f.read()
            self.states = lines.split("\n")
        
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
                # print(payload)
                f.write(",".join(map(str, payload))+"\n")
