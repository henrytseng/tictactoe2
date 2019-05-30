from tictactoe.data import Stats

def test_answer():
    assert True == True
    assert False == False

def test_dimensions():
    stats = Stats(None)
    payload = {'a': 1, 'b': 34, 'c': 2, 'd': 4}
    stats.push(payload)
    assert stats.headers == payload.keys()
    assert stats.rounds_df.shape == (1, 4)

