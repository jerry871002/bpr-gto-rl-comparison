import sys
import random
import numpy as np
from keras.utils import to_categorical as one_hot

from env import SoccerEnv

from agents.MADDPG.maddpg import MADDPG
from agents.common.training_opponent import StationaryOpponent, RandomSwitchOpponent, RLBasedOpponent
from agent.BPR.bpr_op import BPR_OP
from agents.BPR.theory_of_mind import TheoryOfMind

# set environment
env = SoccerEnv(width=5, height=5, goal_size=3)

# set agents
ME = TheoryOfMind(env_width=env.width, env_height=env.height, env_goal_size=env.goal_size)
OP = BPR_OP(env_width=env.width, env_height=env.height, env_goal_size=env.goal_size)

# parameters
EPISODES = 5000

for i in range(EPISODES):
    state = env.reset()
    ball_possession_change = True
    win = 0
    win_rate = 0
    past_win_rate = 0
    done = False
    while not done:
        leftx, lefty, rightx, righty, possession = state
        env.show()
        print()
        if ball_possession_change:
            ME.change_policy(leftx, lefty, possession)
            OP.change_policy(rightx, righty, possession)

        # agent 1 decides its action
        actionME = ME.choose_action(state)

        # agent 2 decides its action
        actionOP = OP.choose_action(state)

        # perform actions on the environment
        done, reward_l, reward_r, state_, actions = env.step(actionME, actionOP)

        # training process of agent 1
        ME.update_zero_order_belief(state, actionOP)
        ME.update_first_order_belief(state, actionOP)
        ME.update_confidence(win_rate, past_win_rate)

        # training process of agent 2
        OP.update_belief(state, actionME)

        state = state_
        if done and reward_l == 10:
            win += 1
            past_win_rate = win_rate
            win_rate = win / i
