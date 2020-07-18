import random
import numpy as np
import pickle

from env import SoccerEnv
from soccer_stat import SoccerStat

""" This file provides a baseline by using two random agents
"""

# set environment
env = SoccerEnv(width=5, height=5, goal_size=3)

# parameters
EPISODES = 5000

# statistic
stat = SoccerStat()

for i in range(EPISODES):
    state = env.reset()
    stat.set_initial_ball(state[4])

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
            print(*stat.get_moving_avg(),
                  file=open('log_files/baseline_reward.txt', 'a'))

# save stats
with open('stats/baseline.pkl', 'wb') as output:
    pickle.dump(stat, output)
