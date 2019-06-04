from tictactoe.players import RandomPlayer
from tictactoe.game import Board

def test_answer():
    assert True == True
    assert False == False

def test_move():
    board = Board()
    random = RandomPlayer()
    random.marker = 'X'
    board[0, 1] = random.marker
    board[0, 2] = random.marker
    board[1, 1] = random.marker
    
    assert random.get_move(board) != (0, 1)
    assert random.get_move(board) != (0, 2)
    assert random.get_move(board) != (1, 1)

