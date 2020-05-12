import numpy as np

L = 0
R = 1
W = 0
WO = 1

class SoccerStat:
    def __init__(self):
        self.reset()

    def reset(self):
        self.reward_history = [[], []]
        # win record
        #   | w/ ball | w/o ball
        # -----------------------
        # L |         |
        # R |         |
        self.win_history = []
        self.win_record = [[0, 0], [0, 0]]
        self.ball = None

    def set_initial_ball(self, ball):
        if ball == L:
            self.ball = L
        elif ball == R:
            self.ball = R
        else:
            print(f'Invalid ball possesion: `{ball}`')

    def add_stat(self, reward_l, reward_r):
        self.reward_history[L].append(reward_l)
        self.reward_history[R].append(reward_r)

        if reward_l > reward_r:
            # left win
            self.win_history.append(L)
            if self.ball == L:
                self.win_record[L][W] += 1
            else:
                self.win_record[L][WO] += 1
        elif reward_r > reward_l:
            # right win
            self.win_history.append(R)
            if self.ball == R:
                self.win_record[R][W] += 1
            else:
                self.win_record[R][WO] += 1
        else:
            print(f'Rewards for left and right are the same: {reward_l}')


    def get_moving_avg(self, length=100):
        return (np.mean(self.reward_history[L][-length:]),
                np.mean(self.reward_history[R][-length:]))

    def get_win_rate(self):
        total_game = np.sum(self.win_record)
        win_left = np.sum(self.win_record[L])
        win_right = np.sum(self.win_record[R])
        return (win_left / total_game, win_right / total_game)
