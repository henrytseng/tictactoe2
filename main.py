from tictactoe.game import Game
from tictactoe.players import InputPlayer
from tictactoe.players import RandomPlayer
from contextlib import contextmanager
import time
import argparse

@contextmanager
def benchmark():
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        elapsed_time = end - start
        print("Elapsed time: {}".format(elapsed_time))

def main(input_x=None, input_o=None):
    player1 = RandomPlayer() if not input_x else InputPlayer()
    player2 = RandomPlayer() if not input_o else InputPlayer()

    print("Starting")
    game = Game()
    game.set_player_x(player1)
    game.set_player_o(player2)

    with benchmark():
        for i in range(10000):
            game.start()
            game.end()

if __name__ == "__main__":
    parse = argparse.ArgumentParser(description='Play Tic-Tac-Toe')
    parse.add_argument('-ix', '--input_x', action='store_true', default=False, help='Input based player X')
    parse.add_argument('-io', '--input_o', action='store_true', default=False, help='Input based player O')

    args = parse.parse_args()
    main(**vars(args))
