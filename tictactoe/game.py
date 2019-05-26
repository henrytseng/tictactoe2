import random
import time

class Board(object):

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
            irows = (self.__getitem__((i, j)) for i in range(self.width) if self.__getitem__((i, j)) is not None)
            irows_set = set(irows)
            if len(irows_set) == 1:
                return list(irows_set)[0]
        
        # Check columns
        for i in range(self.width):
            jcols = (self.__getitem__((i, j)) for j in range(self.height) if self.__getitem__((i, j)) is not None)
            jcols_set = set(jcols)
            if len(jcols_set) == 1:
                return list(jcols_set)[0]

        # Left diagnol
        ldiag = (self.__getitem__((i, i)) for i in range(self.width) if self.__getitem__((i, j)) is not None)
        ldiag_set = set(ldiag)
        if len(ldiag_set) == 1:
            return list(ldiag_set)[0]
        
        # Right diagnol
        rdiag = (self.__getitem__((self.width-1-i, i)) for i in range(self.width) if self.__getitem__((i, j)) is not None)
        rdiag_set = set(rdiag)
        if len(rdiag_set) == 1:
            return list(rdiag_set)[0]

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
        return self.serialize(empty="-", row_sep="\n", col_sep="")

    def serialize(self, empty="-", row_sep="", col_sep=""):
        rows = []
        for j in range(self.height):
            cols = []
            for i in range(self.width):
                item = self.__getitem__((i, j))
                if item is None:
                    item = empty
                cols.append(item)
            rows.append(col_sep.join(cols))
        return row_sep.join(rows)

    def deserialize(self, state, empty="-", row_sep="", col_sep=""):
        # TODO implement
        return


class Game(object):

    def __init__(self):
        # self.saved_state = None
        self.board = Board()

    def set_player_x(self, player):
        self.player_x = player

    def set_player_o(self, player):
        self.player_o = player

    def reset(self):
        self.id = hex(random.getrandbits(16))
        if self.saved_state is not None:
            self.saved_state.reset()
        self.winner = None
        self.board.reset()

    def start(self):
        self.reset()
        print("Starting {}".format(self.id))
        for i in range(self.board.width * self.board.height):
            self.update(i)
            if self.winner is not None:
                # print("Winner: {}".format(n))
                return True
        # print("Play again")
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
            if False:
                pass
            else:
                print(self.board.serialize())

        self.winner = self.board.get_winner()
        return self.winner

    def end(self):
        print("Complete {}".format(self.id))

