import sys
import random
import numpy as np
from keras.utils import to_categorical as one_hot

from env import SoccerEnv

from agents.MADDPG.maddpg import MADDPG
from agents.common.training_opponent import StationaryOpponent, RandomSwitchOpponent, RLBasedOpponent

# set environment
env = SoccerEnv(width=5, height=5, goal_size=3)

# set agents
ME = "add your agent here"
OP = StationaryOpponent(env_width=env.width, env_height=env.height, env_goal_size=env.goal_size)

# parameters
EPISODES = 5000

for i in range(EPISODES):
    state = env.reset()

    done = False
    while not done:
        env.show()
        print()

        # agent 1 decides its action

        # agent 2 decides its action
        actionOP = OP.get_action(state)

        # perform actions on the environment
        done, reward_l, reward_r, state_, actions = env.step(actionME, actionOP)

        # training process of agent 1


        # training process of agent 2
        OP.adjust(done, reward_r, i)

        state = state_
