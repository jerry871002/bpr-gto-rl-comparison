import sys
import random
import numpy as np
from keras.utils import to_categorical as one_hot
import matplotlib.pyplot as plt
from env import SoccerEnv
from agents.BPR.bpr import BPR
from agents.common.training_opponent import StationaryOpponent, RandomSwitchOpponent, RLBasedOpponent

# set environment
env = SoccerEnv(width=5, height=5, goal_size=3)

# set agents
ME = BPR(env_width=env.width, env_height=env.height, env_goal_size=env.goal_size)
OP = RandomSwitchOpponent(env_width=env.width, env_height=env.height, \
                          env_goal_size=env.goal_size, episode_reset=20)

# parameters
EPISODES = 100
win = 0
belief_attack_change = []
belief_defense_change = []
op_type = []
for i in range(EPISODES):
    state = env.reset()
    ball_possession_change = True
    reward = 0
    done = False
    op_type.append(OP.type_attack)
    while not done:
        env.show()
        print()
        # agent 1 decides its action
        if ball_possession_change:
            ME.change_policy(state[0], state[1], state[4])
        actionME = ME.choose_action(state)
        # agent 2 decides its action
        actionOP = OP.get_action(state)

        # perform actions on the environment
        done, reward_l, reward_r, state_, actions = env.step(actionME, actionOP)

        # training process of agent 1
        ME.update_belief(state, actions[1])  # update every episode or every step?????

        # training process of agent 2
        OP.adjust(done, reward_r, i)
        ball_possession_change = not(state[4] == state_[4])

        state = state_
        reward += reward_l

        if done and reward_l == 10:
            win += 1
    belief_attack_change.append(ME.belief_attack)
    belief_defense_change.append(ME.belief_defense)
print('win rate: ', win / EPISODES)

plt.plot(belief_attack_change)
plt.legend(['type1', 'type2', 'type3', 'type4', 'type5'], loc='center right')
plt.xlabel('Episodes')
plt.ylabel('Belief')
