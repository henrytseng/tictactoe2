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
                    if header is None:
                        header = item.keys()
                        writer.writerow(header)
                    elif item == 'complete':
                        exit()
                    else:
                        logging.info('Getting: {}'.format(item.values()))
                        writer.writerow(item.values())
        except Exception as e:
            logger.error(traceback.format_exc())
        finally:
            self.lock.release()


class Stats(object):

    def __init__(self, queue=None, width=None, height=None):
        self.queue = queue
        self.width = width
        self.height = height
        self.reset()

    def load(self, src):
        '''Loads recorded games'''
        self.rounds_df = pd.read_csv(src, header=None, names=None)
        self.rounds_df = self.rounds_df.fillna('NA')

        with open(src, 'r', newline='') as f:
            reader = csv.reader(f, quoting=csv.QUOTE_MINIMAL)
            for i in reader:
                logger.info(i)

    def push(self, data):
        if self.rounds_df is None:
            self.headers = data.keys()
            self.rounds_df = pd.DataFrame(columns=list(data.keys()))
        self.rounds_df = self.rounds_df.append(data, ignore_index=True)

    def complete(self, data):
        # Build data
        for k, v in data.items():
            self.rounds_df[k] = v
        self.headers = self.rounds_df.columns
        
        # Send
        for i, row in self.rounds_df.iterrows():
            if self.queue is not None:
                self.queue.put(dict(row))
                logger.info('Sending: {}'.format(tuple(row)))
        self.reset()

    def reset(self):
        self.rounds_df = None
