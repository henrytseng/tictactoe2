from os import environ
import traceback
import random
import logging
import csv
from multiprocessing import get_logger
import pandas as pd
import numpy as np

logger = get_logger()

class Storage(object):

    def __init__(self, queue, lock):
        self.queue = queue
        self.lock = lock
        self.temp_path = environ['DATA_FOLDER'] if 'DATA_FOLDER' in environ else './data'
        self.filename = "{}.csv".format(hex(random.getrandbits(16)))

    def collect(self):
        file_path = "{temp_path}/{filename}".format(temp_path=self.temp_path, filename=self.filename)
        logger.info("Collecting {file}".format(file=file_path))
        try:
            self.lock.acquire()
            with open(file_path, 'w+', newline='') as f:
                writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                header = None
                for item in iter(self.queue.get, None):
                    if item[0] == 'header':
                        writer.writerow(item[1])
                    elif item[0] == 'complete':
                        exit()
                    elif item[0] == 'move':
                        logger.debug('Getting: {}'.format(item[1].values()))
                        writer.writerow(item[1].values())
        except Exception as e:
            logger.error(traceback.format_exc())
        finally:
            self.lock.release()


class Stats(object):

    def __init__(self, queue=None, width=None, height=None):
        self.queue = queue
        self.width = width
        self.height = height
        self.rounds_df = None
        self.metrics = { 'X':0, 'O':0, 'total': 0 }
        self.reset()

    def push(self, data):
        if self.rounds_df is None:
            self.rounds_df = pd.DataFrame(columns=list(data.keys()))
        self.rounds_df = self.rounds_df.append(data, ignore_index=True)

    def complete(self, data):
        # Build data
        for k, v in data.items():
            self.rounds_df[k] = v
        self.headers = self.rounds_df.columns
        
        # Send
        if self.queue is not None:
            self.queue.put(('header', self.headers))
        for i, row in self.rounds_df.iterrows():
            if self.queue is not None:
                self.queue.put(('move', dict(row)))
                logger.debug('Sending: {}'.format(tuple(row)))
        self.reset()

        # Stats
        marker = data['winner']
        if marker not in self.metrics:
            self.metrics[marker] = 0
        self.metrics[marker] += 1 
        self.metrics['total'] += 1
        self.metrics['rx'] = self.metrics['X'] / self.metrics['total']
        self.metrics['ro'] = self.metrics['O'] / self.metrics['total']
        logger.info("Game stats {}".format(self.metrics))

    def reset(self):
        self.rounds_df = None
