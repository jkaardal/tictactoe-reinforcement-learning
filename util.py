import numpy as np
from .game import TicTacToe


def get_all_analogues(game):
    analogues = []

    for i in range(4):
        analogue = TicTacToe()
        for j in range(1, len(game.state_history)):
            analogue.state_history.append(np.rot90(np.copy(game.state_history[j]), i))
        for a in game.action_history:
            row, col = TicTacToe.rot90(a[0], a[1], i)
            analogue.action_history.append([row, col])
        analogue.state = np.copy(analogue.state_history[-1])
        if len(analogue.action_history) % 2 == 0:
            analogue.current_player = 1
        else:
            analogue.current_player = -1
        analogues.append(analogue)

    for i in range(4):
        analogue = TicTacToe()
        for j in range(1, len(game.state_history)):
            analogue.state_history.append(np.copy(np.rot90(game.state_history[j][:, ::-1], i)))
        for a in game.action_history:
            row, col = TicTacToe.rot90(a[0], 2 - a[1], i)
            analogue.action_history.append([row, col])
        analogue.state = np.copy(analogue.state_history[-1])
        if len(analogue.action_history) % 2 == 0:
            analogue.current_player = 1
        else:
            analogue.current_player = -1
        analogues.append(analogue)
    
    return analogues