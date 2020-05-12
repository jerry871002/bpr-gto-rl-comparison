import random
import numpy as np

from env import SoccerEnv

""" This file provides a baseline by using two random agents
"""

# set environment
env = SoccerEnv(width=5, height=5, goal_size=3)

# parameters
EPISODES = 5000

# record rewards
rewardL_history = []
rewardR_history = []

# record win rate
winL = 0
winR = 0

for i in range(EPISODES):
    state = env.reset()

    rewardL = 0
    rewardR = 0
    done = False
    while not done:
        # env.show()
        # print()

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
            rewardL_history.append(rewardL)
            rewardR_history.append(rewardR)
            print(np.mean(rewardL_history[-100:]),
                  np.mean(rewardR_history[-100:]),
                  file=open('log_files/baseline_reward.txt', 'a'))

            if rewardL > rewardR:
                winL += 1
            else:
                winR += 1
            print(f'{winL / (i+1)} {winR / (i+1)}',
                  file=open('log_files/baseline_win_rate.txt', 'a'))
