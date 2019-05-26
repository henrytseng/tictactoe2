import traceback
from contextlib import contextmanager
from multiprocessing import Pool, Lock, log_to_stderr, get_logger
import argparse
import os
import time
from tictactoe.game import Game, Board
from tictactoe.players import InputPlayer
from tictactoe.players import RandomPlayer

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
        print("Elapsed time: {time}".format(time=elapsed_time))

def play(lock=None):
    print("Running {}".format(os.getpid()))
    try:
        lock.acquire()
        # player1 = RandomPlayer() if not kwargs['input_x'] else InputPlayer()
        # player2 = RandomPlayer() if not kwargs['input_o'] else InputPlayer()
        # game = Game()
        # game.set_player_x(player1)
        # game.set_player_o(player2)
        # game.start()
        # game.end()
    except Exception as e:
        get_logger().error(traceback.format_exc())
        raise
    finally:
        lock.release()
    return True

def main(**kwargs):
    lock = Lock()
    log_to_stderr()
    with benchmark():
        game_pool = Pool(4)
        # for i in range(10):
        #     results = game_pool.apply_async(play, kwds={lock:lock})
        game_pool.close()
        game_pool.join()

# Entry point
if __name__ == "__main__":
    parse = argparse.ArgumentParser(description='Play Tic-Tac-Toe')
    parse.add_argument('-ix', '--input_x', action='store_true', default=False, help='Input based player X')
    parse.add_argument('-io', '--input_o', action='store_true', default=False, help='Input based player O')

    args = parse.parse_args()
    main(**vars(args))
