import numpy as np


class InvalidActionError(Exception):
    pass


class TicTacToe:

    def __init__(self):
        self.state = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.state_history = [np.copy(self.state)]
        self.action_history = []

    @staticmethod
    def unroll(row, col):
        return row * 3 + col

    @staticmethod
    def roll(index):
        row = index // 3
        col = index % 3
        return row, col

    @staticmethod
    def action_to_roll(action):
        row, col = np.where(action != 0)
        if len(row) and len(col):
            return row[0], col[0]
        else:
            return None
        
    @staticmethod
    def roll_to_action(row, col, val):
        action = np.zeros((3, 3), dtype=int)
        action[row, col] = val
        return action

    @staticmethod
    def key_to_roll(key):
        row, col = TicTacToe.roll(key - 1)
        return 2 - row, col
    
    @staticmethod
    def rot90(row, col, k):
        index = TicTacToe.unroll(row, col)
        M = np.arange(9, dtype=int).reshape(3, 3)
        M = np.rot90(M, k)
        rows, cols = np.where(M == index)
        return rows[0], cols[0]

    def move(self, row, col):
        # add a symbol to the game board
        assert row >= 0 and row < 3
        assert col >= 0 and row < 3

        if self.state[row, col] != 0:
            raise InvalidActionError(f'Invalid move! Position {row}, {col} is already occupied!')
        elif self.check_termination() is not None:
            raise InvalidActionError(f'Game is already complete! Must call reset() to start a new game.')

        self.state[row, col] = self.current_player
        if self.current_player == 1:
            self.current_player = -1
        else:
            self.current_player = 1
        self.action_history.append([row, col])
        self.state_history.append(np.copy(self.state))

    def check_termination(self):
        # check whether terminal state has been reached and the outcome
        mask_1 = self.state == 1
        if (np.any(np.prod(mask_1, axis=1)) or np.any(np.prod(mask_1, axis=0)) or
              np.prod(mask_1[np.eye(3, dtype=bool)]) or np.prod(mask_1[:, ::-1][np.eye(3, dtype=bool)])):
            return 1

        mask_2 = self.state == -1
        if (np.any(np.prod(mask_2, axis=1)) or np.any(np.prod(mask_2, axis=0)) or
              np.prod(mask_2[np.eye(3, dtype=bool)]) or np.prod(mask_2[:, ::-1][np.eye(3, dtype=bool)])):
            return -1

        if np.sum(self.state == 0) > 0:
            return None
        else:
            return 0

    def valid_action_coordinates(self):
        # return an array of row/column pairs
        row, col = np.where(self.state == 0)
        return np.concatenate([row.reshape(-1, 1), col.reshape(-1, 1)], axis=1)

    def valid_actions(self):
        # return a list of valid actions given the current state
        row, col = np.where(self.state == 0)
        actions = []
        for r, c in zip(row, col):
            a = np.zeros((3, 3), dtype=int)
            a[r, c] = self.current_player
            actions.append(a)
        return actions

    def __str__(self):
        # current state to string
        rows = []
        for i in range(3):
            msg = []
            for j in range(3):
                if self.state[i, j] == 1:
                    msg.append('X')
                elif self.state[i, j] == -1:
                    msg.append('O')
                else:
                    msg.append(' ')
            msg = '|'.join(msg) + '\n'
            rows.append(msg)
        return '-----\n'.join(rows)

    def print_state(self):
        # pretty print the current state
        print(self.__str__())

    def reset(self):
        # reset the game to an empty board
        self.state = np.zeros((3, 3), dtype=int)
        self.state_history = [np.copy(self.state)]
        self.action_history = []
        self.current_player = 1
  