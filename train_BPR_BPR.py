import sys
import random
import numpy as np
from keras.utils import to_categorical as one_hot

from env import SoccerEnv

from agents.BPR.bpr import BPR
from agents.BPR.bpr_op import BPR_OP
from agents.common.training_opponent import StationaryOpponent, RandomSwitchOpponent, RLBasedOpponent
from agents.common.common_agents import StationaryAgent, RandomSwitchAgent, RLBasedAgent

# set environment
env = SoccerEnv(width=5, height=5, goal_size=3)

# set agents
ME = BPR(env_width=env.width, env_height=env.height, env_goal_size=env.goal_size)
# ME = RandomSwitchAgent(env_width=env.width, env_height=env.height, env_goal_size=env.goal_size, episode_reset=15)
OP = BPR_OP(env_width=env.width, env_height=env.height, env_goal_size=env.goal_size)

# parameters
EPISODES = 500
win = 0
for i in range(EPISODES):
    state = env.reset()
    ball_possession_change = True
    reward = 0
    done = False
    while not done:
        leftx, lefty, rightx, righty, possession = state
        env.show()
        print()
        # input('wait')
        # agent 1 decides its action
        if ball_possession_change:
            ME.change_policy(leftx, lefty, possession)
            OP.change_policy(rightx, righty, possession)
        actionME = ME.choose_action(state)
        # agent 2 decides its action
        actionOP = OP.choose_action(state)
        # print(f'actionOP:{actionOP}')

        # perform actions on the environment
        done, reward_l, reward_r, state_, actions = env.step(actionME, actionOP)
        
        if done and reward_l==10:
            win+=1
        
        # training process of agent 1
        # ME.update_belief(state, actions[1]) #update every episode or every step?????
        ME.update_belief(state, actions[1])
        # training process of agent 2
        OP.update_belief(state, actions[0])
        
        ball_possession_change = not(state[4]==state_[4])
        state = state_

print(f'win rate = {win/EPISODES}')