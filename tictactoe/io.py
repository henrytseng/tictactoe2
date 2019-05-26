from os import environ
import random
import multiprocessing

class StateQueue(object):

    def __init__(self):
        pass


class StateWriter(object):

    def __init__(self):
        self.temp_path = environ['DATA_FOLDER'] if 'DATA_FOLDER' in environ else './data'
        self.filename = "{}.data".format(hex(random.getrandbits(16)))
        self.rounds = []

    def reset(self):
        self.rounds = []

    def push(self, ts=None, board=None, player=None, move=None):
        turn = (
            ts,
            player,
            board.serialize(),
            move,
        )
        self.rounds.append(turn)

    def store(self, id, winner):
        rounds_list = self.rounds
        self.rounds = []
        with open("{tmp}/{filename}".format(tmp=self.temp_path, filename=self.filename), 'a') as f:
            for i in rounds_list:
                payload = (
                    id,
                    winner,
                ) + i
                print(payload)
                f.write(",".join(map(str, payload))+"\n")


