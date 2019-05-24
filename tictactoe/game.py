from .io import SavedState

class Board(object):

    def __init__(self, width=3, height=3):
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        '''Builds an empty game board'''
        self.places = [None for i in range(self.height * self.width)]
        return self.places

    def find_empty(self):
        '''Gets a list of empty spaces'''
        return [(i, j) for j in range(self.height) for i in range(self.width) if self.places[j * self.width + i] is None]

    def has_empty(self):
        return None in self.places

    def get_winner(self):
        # Check rows
        for j in range(self.height):
            irows = (self.__getitem__((i, j)) for i in range(self.width))
            irows_set = set(irows)
            if len(irows_set) == 1:
                return next(iter(irows_set))
        
        # Check columns
        for i in range(self.width):
            jcols = (self.__getitem__((i, j)) for j in range(self.height))
            jcols_set = set(jcols)
            if len(jcols_set) == 1:
                return next(iter(jcols_set))

        # Left diagnol
        ldiag = (self.__getitem__((i, i)) for i in range(self.width))
        ldiag_set = set(ldiag)
        if len(ldiag_set) == 1:
            return next(iter(ldiag_set))
        
        # Right diagnol
        rdiag = (self.__getitem__((self.width-1-i, i)) for i in range(self.width))
        rdiag_set = set(rdiag)
        if len(rdiag_set) == 1:
            return next(iter(rdiag_set))

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
        return


class Game(object):

    def __init__(self):
        self.board = Board()
        self.saved_state = SavedState()

    def set_player_x(self, player):
        self.player_x = player

    def set_player_o(self, player):
        self.player_o = player

    def reset(self):
        self.board.reset()
        self.winner = None

    def start(self):
        for i in range(self.board.width * self.board.height):
            n = self.update(i)
            if n:
                # print("Winner: {}".format(n))
                return True
        # print("Play again")
        return False

    def update(self, i):
        if i % 2 == 0:
            player = self.player_x
            marker = 'X'
        else:
            player = self.player_o
            marker = 'O'
        if(self.board.has_empty()):
            move = player.get_move(self.board)
            self.board[move] = marker
            self.saved_state.push(board=self.board, player=marker, move=move)

        self.winner = self.board.get_winner()
        if self.winner:
            self.end()

        # print(self.board.serialize())
        return self.winner

    def end(self):
        self.saved_state.store(winner=self.winner)
        self.saved_state.reset()
        self.reset()

