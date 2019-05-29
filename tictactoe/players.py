from .game import Board
import random
import logging

logger = logging.getLogger(__name__)

class RandomPlayer(object):

    def get_move(self, board):
        return random.choice(board.find_empty())


class LearningPlayer(object):

    def get_move(self, board):
        return random.choice(board.find_empty())


class InputPlayer(object):

    def get_move(self, board):
        print(board.debug())
        print("Enter a coordinate (i,j):")
        value = input()
        if value == "":
            print("Exiting")
            exit()
        return list(map(lambda x:int(x), value.split(',')))
