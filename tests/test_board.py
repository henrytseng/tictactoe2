from tictactoe.game import Board

def test_answer():
    assert True == True
    assert False == False

def test_dimensions():
    board = Board()
    assert board.width == 3
    assert board.height == 3
    assert len(board.places) == board.width * board.height

def test_placement():
    board = Board()
    assert board.has_empty() == True
    assert len(board.find_empty()) == len(board.places)
    assert board.get_winner() == None

    # Take a spot
    board[0, 0] = 'X'
    assert board[0, 0] == 'X'
    assert len(board.find_empty()) == len(board.places) - 1
    board.reset()
    assert len(board.find_empty()) == board.width * board.height

def test_location():
    board = Board()
    for i in range(board.width):
        for j in range(board.height):
            board[i, j] = 'X'
            assert board[i, j] == 'X'

def test_winner():
    board = Board()

    # Top column
    board.reset()
    board[0, 0] = 'X'
    board[0, 1] = 'X'
    board[0, 2] = 'X'
    assert board.get_winner() == 'X'

    # Mid column
    board.reset()
    board[1, 0] = 'X'
    board[1, 1] = 'X'
    board[1, 2] = 'X'
    assert board.get_winner() == 'X'

    # Top column
    board.reset()
    board[2, 0] = 'X'
    board[2, 1] = 'X'
    board[2, 2] = 'X'
    assert board.get_winner() == 'X'

    # Left column
    board.reset()
    board[0, 0] = 'X'
    board[1, 0] = 'X'
    board[2, 0] = 'X'
    assert board.get_winner() == 'X'

    # Mid column
    board.reset()
    board[0, 1] = 'X'
    board[1, 1] = 'X'
    board[2, 1] = 'X'
    assert board.get_winner() == 'X'

    # Right column
    board.reset()
    board[0, 2] = 'X'
    board[1, 2] = 'X'
    board[2, 2] = 'X'
    assert board.get_winner() == 'X'
