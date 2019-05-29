from os import environ
import random
import time
import math
import logging
from .data import Stats

logger = logging.getLogger(__name__)

class Board(object):

    @classmethod
    def serialize(cls, board, empty="-", row_sep="", col_sep=""):
        rows = []
        for j in range(board.height):
            cols = []
            for i in range(board.width):
                item = board.__getitem__((i, j))
                if item is None:
                    item = empty
                cols.append(item)
            rows.append(col_sep.join(cols))
        return row_sep.join(rows)

    @classmethod
    def deserialize(cls, serialized, empty="-", row_sep="", col_sep=""):
        dim = int(math.sqrt(len(serialized)))
        board = Board(dim, dim)
        for j in range(dim):
            for i in range(dim):
                board[i, j] = serialized[j * dim + i]
        return board

    def __init__(self, width=3, height=3):
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        '''Builds an empty game board'''
        self.places = [None for i in range(self.height * self.width)]

    def find_empty(self):
        '''Gets a list of empty spaces'''
        return [(i, j) for j in range(self.height) for i in range(self.width) if self.places[j * self.width + i] is None]

    def has_empty(self):
        return None in self.places

    def get_winner(self):
        # Check rows
        for j in range(self.height):
            irows = [self.__getitem__((i, j)) for i in range(self.width) if self.__getitem__((i, j)) is not None]
            if len(set(irows)) == 1 and len(irows) == self.width:
                return irows[0]
        
        # Check columns
        for i in range(self.width):
            jcols = [self.__getitem__((i, j)) for j in range(self.height) if self.__getitem__((i, j)) is not None]
            if len(set(jcols)) == 1 and len(jcols) == self.height:
                return jcols[0]

        # Left diagnol
        ldiag = [self.__getitem__((i, i)) for i in range(self.width) if self.__getitem__((i, i)) is not None]
        if len(set(ldiag)) == 1 and len(ldiag) == self.width:
            return ldiag[0]
        
        # Right diagnol
        rdiag = [self.__getitem__((self.width-1-i, i)) for i in range(self.width) if self.__getitem__((self.width-1-i, i)) is not None]
        if len(set(rdiag)) == 1 and len(rdiag) == self.width:
            return rdiag[0]

        return None

    def __getitem__(self, key):
        return self.places[key[1] * self.width + key[0]]

    def __setitem__(self, key, value):
        n = key[1] * self.width + key[0]
        if self.places[n] != None:
            raise Exception('Place already taken')
        self.places[n] = value

    def debug(self):
        '''Debugging display'''
        return Board.serialize(self, empty="-", row_sep="\n", col_sep="")


class Game(object):

    def __init__(self, queue):
        self.debug_level = environ['DEBUG_LEVEL'] if 'DEBUG_LEVEL' in environ else None
        self.board = Board()
        self.stats = Stats(queue)
        self.reset()

    def set_player_x(self, player):
        self.player_x = player

    def set_player_o(self, player):
        self.player_o = player

    def reset(self):
        self.id = hex(random.getrandbits(16))
        self.winner = None
        self.board.reset()

    def start(self):
        self.reset()
        if self.debug_level == 'info':
            logger.info("Starting {}".format(self.id))
        for i in range(self.board.width * self.board.height):
            self.update(i)
            if self.winner is not None:
                # logger.info("Winner: {}".format(self.winner))
                return True
        # logger.info("Play again")
        return False

    def update(self, i):
        if self.winner:
            raise Exception("Error winner {} already selected.")
        if i % 2 == 0:
            player = self.player_x
            marker = 'X'
        else:
            player = self.player_o
            marker = 'O'
        if(self.board.has_empty()):
            move = player.get_move(self.board)
            self.board[move] = marker
            self.stats.push(tuple(marker) + move + tuple(Board.serialize(self.board)))
        self.winner = self.board.get_winner()
        return self.winner

    def end(self):
        self.stats.complete((self.id, self.winner))
        if self.debug_level == 'info':
            logger.info("Complete {}".format(self.id))

