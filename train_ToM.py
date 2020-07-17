import sys
import random
import numpy as np
from keras.utils import to_categorical as one_hot

from env import SoccerEnv

from agents.MADDPG.maddpg import MADDPG
from agents.common.training_opponent import StationaryOpponent, RandomSwitchOpponent, RLBasedOpponent
from agents.BPR.bpr_op import BPR_OP
from agents.BPR.theory_of_mind import TheoryOfMind

import matplotlib.pyplot as plt

# set environment
env = SoccerEnv(width=5, height=5, goal_size=3)

# set agents
ME = TheoryOfMind(env_width=env.width, env_height=env.height, env_goal_size=env.goal_size)
# OP = RandomSwitchOpponent(env_width=env.width, env_height=env.height, env_goal_size=env.goal_size, episode_reset=50)
# OP = StationaryOpponent(env_width=env.width, env_height=env.height, env_goal_size=env.goal_size, type_attack=4, type_defense=0)
OP = BPR_OP(env_width=env.width, env_height=env.height, env_goal_size=env.goal_size)
# parameters
EPISODES = 200
win = 0
win_rate = 0
past_win_rate = 0
# belief_attack_change = []
belief_defense_change = []
# op_type = []
for i in range(EPISODES):
    state = env.reset()
    ball_possession_change = True
    done = False
    # op_type.append(OP.policy_attack)
    while not done:
        leftx, lefty, rightx, righty, possession = state
        env.show()
        print()
        print(i+1)
        # input('wait')
        
        if ball_possession_change:
            ME.change_policy(leftx, lefty, possession)
            OP.change_policy(rightx, righty, possession) # function of BPR_OP

        # agent 1 decides its action
        actionME = ME.choose_action(state)

        # agent 2 decides its action
        actionOP = OP.choose_action(state) # function of BPR_OP
        # actionOP = OP.get_action(state)

        # perform actions on the environment
        done, reward_l, reward_r, state_, actions = env.step(actionME, actionOP)

        # training process of agent 1
        ME.update_zero_order_belief(state, actionOP)
        ME.update_first_order_belief(state, actionOP)

        # training process of agent 2
        OP.update_belief(state, actionME) # function of BPR_OP
        # OP.adjust(done, reward_r, i)

        ball_possession_change = not(state[4]==state_[4])
        state = state_
        if done and reward_l == 10:
            print('win')
            win += 1
        elif done and reward_l == -10:
            print('lose')
    past_win_rate = win_rate
    win_rate = win / (i+1)
    ME.update_confidence(win_rate, past_win_rate)
    print(f'current win rate = {win_rate}')
    print(f'confidence = {ME.confidence}')
    # belief_attack_change.append(ME.zero_belief_attack)
    belief_defense_change.append(ME.zero_belief_defense)
print(f'win rate = {win/EPISODES}')
# plt.plot(belief_defense_change)
# plt.legend(['type1', 'type2', 'type3', 'type4', 'type5'])
