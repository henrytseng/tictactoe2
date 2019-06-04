import traceback
from contextlib import contextmanager
from multiprocessing import Pool, Process, Queue, Lock, log_to_stderr
import argparse
import os
import time
import logging
from tictactoe.game import Game
from tictactoe.players import InputPlayer, RandomPlayer, LearningPlayer
from tictactoe.data import Storage

logger = logging.getLogger(__name__)
logger_subprocess = log_to_stderr()
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

def get_player(player_type, learning_file):
    player_map = {
        'random': RandomPlayer(),
        'input': InputPlayer(),
        'learning': LearningPlayer(learning_file)
    }
    return player_map[player_type]

def play(player1=None, player2=None, learning_file=None):
    logger_subprocess.info("Running {}".format(os.getpid()))
    try:
        game = Game(queue)
        game.set_player_x(get_player(player1, learning_file))
        game.set_player_o(get_player(player2, learning_file))
        game.start()
        game.end()
    except Exception as e:
        logger_subprocess.error(traceback.format_exc())
        raise
    return True

def store(q, l):
    try:
        storage = Storage(q, l)
        storage.collect()
    except Exception as e:
        logger_subprocess.error(traceback.format_exc())
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
    learning_file = kwargs['learning_file']
    
    with benchmark():
        # Set pool size to number of processors
        pool = Pool(kwargs['num_concurrency'], initializer=init, initargs=(l, q))
        for i in range(kwargs['num_games']):
            pool.apply_async(play, args=(player1, player2, learning_file))
        pool.close()
        pool.join()
        q.put('complete')
    p.join()

# Entry point
if __name__ == "__main__":
    parse = argparse.ArgumentParser(description='Play Tic-Tac-Toe')
    parse.add_argument('-ix', '--input_x', default='random', help='Input based player X')
    parse.add_argument('-io', '--input_o', default='learning', help='Input based player O')
    parse.add_argument('-n', '--num_games', type=int, default=2, help='Number of games to play')
    parse.add_argument('-c', '--num_concurrency', type=int, default=1, help='Number of games to play concurrently')
    parse.add_argument("-v", "--verbose", default=1, action="count", help="Increase logging verbosity")
    parse.add_argument("-f", "--learning_file", help="Loads learning data from historical games")

    args = parse.parse_args()
    main(**vars(args))
