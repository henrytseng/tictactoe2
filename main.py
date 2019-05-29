import traceback
from contextlib import contextmanager
from multiprocessing import Pool, Process, Queue, Lock, log_to_stderr
import argparse
import os
import time
import logging
from tictactoe.game import Game, Board
from tictactoe.players import InputPlayer, RandomPlayer, LearningPlayer
from tictactoe.data import Storage

logger = logging.getLogger(__name__)
lock = None
queue = None

@contextmanager
def benchmark(data={}):
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        elapsed_time = end - start
        data['start'] = start
        data['end'] = end
        data['elapsed'] = elapsed_time
        logger.info("Elapsed time: {time}".format(time=elapsed_time))

def init(l, q):
    global lock, queue
    lock = l
    queue = q

def get_player(arg):
    player_map = {
        'random': RandomPlayer(),
        'input': InputPlayer(),
        'learning': LearningPlayer()
    }
    return player_map[arg]

def play(player1=None, player2=None):
    logger = log_to_stderr()
    logger.info("Running {}".format(os.getpid()))
    try:
        game = Game(queue)
        game.set_player_x(get_player(player1))
        game.set_player_o(get_player(player2))
        game.start()
        game.end()
    except Exception as e:
        logger.error(traceback.format_exc())
        raise
    return True

def store(q, l):
    logger = log_to_stderr()
    try:
        storage = Storage(q, l)
        storage.collect()
    except Exception as e:
        logger.error(traceback.format_exc())
        raise

def main(**kwargs):
    # Setup logging
    logging_level = {
        2: logging.DEBUG,
        1: logging.INFO,
        0: logging.WARNING,
    }[min(2, kwargs['verbose'])]
    logging.basicConfig(level=logging_level)

    l = Lock()
    q = Queue()
    p = Process(target=store, args=(q, l))
    p.start()

    player1 = kwargs['input_x']
    player2 = kwargs['input_o']
    
    with benchmark():
        pool = Pool(2, initializer=init, initargs=(l, q))
        for i in range(kwargs['num_games']):
            pool.apply_async(play, args=(player1, player2))
        pool.close()
        pool.join()
        q.put('complete')
    p.join()

# Entry point
if __name__ == "__main__":
    parse = argparse.ArgumentParser(description='Play Tic-Tac-Toe')
    parse.add_argument('-ix', '--input_x', default='random', help='Input based player X')
    parse.add_argument('-io', '--input_o', default='learning', help='Input based player O')
    parse.add_argument('-n', '--num_games', type=int, default=1, help='Number of games to play')
    parse.add_argument("-v", "--verbose", default=0, action="count", help="Increase logging verbosity")

    args = parse.parse_args()
    main(**vars(args))
