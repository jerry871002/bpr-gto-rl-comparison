import random
import numpy as np

from env import SoccerEnv
from soccer_stat import SoccerStat

# set environment
env = SoccerEnv(width=5, height=5, goal_size=3)
# statistic
stat = SoccerStat()

# parameters
EPISODES = 100

stat.reset()

for i in range(EPISODES):
    state = env.reset()
    stat.set_initial_ball(state[4]) # this is essential!!!

    rewardL = 0
    rewardR = 0
    done = False
    while not done:
        # agent 1 decides its action
        actionL = random.randint(0, env.act_dim-1)

        # agent 2 decides its action
        actionR = random.randint(0, env.act_dim-1)

        # perform actions on the environment
        done, reward_l, reward_r, state_, actions = env.step(actionL, actionR)

        state = state_

        rewardL += reward_l
        rewardR += reward_r

        if done:
            stat.add_stat(rewardL, rewardR)
            print(*stat.get_moving_avg())

print(stat.win_record) # [[25, 22], [28, 25]]
print(stat.get_moving_avg()) # (-12.86, -11.28)
print(stat.get_win_rate()) # (0.47, 0.53)
print(stat.win_history) # [1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0]
