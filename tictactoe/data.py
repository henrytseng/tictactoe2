from os import environ
import traceback
import random
import logging
import csv
from multiprocessing import log_to_stderr
import numpy as np

logger = log_to_stderr()

class Storage(object):

    def __init__(self, queue, lock):
        self.queue = queue
        self.lock = lock
        self.temp_path = environ['DATA_FOLDER'] if 'DATA_FOLDER' in environ else './data'
        self.filename = "{}.csv".format(hex(random.getrandbits(16)))

    def collect(self):
        try:
            self.lock.acquire()
            with open("{temp_path}/{filename}".format(temp_path=self.temp_path, filename=self.filename), 'w+', newline='') as f:
                writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                for item in iter(self.queue.get, None):
                    if item == 'complete':
                        exit()
                    logging.info('Getting: {}'.format(item))
                    writer.writerow(item)
        except Exception as e:
            logger.error(traceback.format_exc())
        finally:
            self.lock.release()

    # def store(self, id, rounds, winner):
    #     rounds_list = self.rounds
    #     with open("{tmp}/{filename}".format(tmp=self.temp_path, filename=self.filename), 'a') as f:
    #         for i in rounds_list:
    #             payload = (
    #                 id,
    #                 winner,
    #             ) + i
    #             logger.info(payload)
    #             f.write(",".join(map(str, payload))+"\n")

    def reset():
        self.data = []


class Stats(object):

    def __init__(self, queue):
        self.queue = queue
        self.reset()

    def push(self, data):
        self.rounds.append(data)

    def complete(self, data):
        for i in self.rounds:
            item = tuple(data) + i
            self.queue.put(item)
            logger.info('Sending: {}'.format(item))
        self.reset()

    def reset(self):
        self.rounds = []
