import sys
import pickle
import numpy as np
from keras.utils import to_categorical as one_hot

from env import SoccerEnv
from soccer_stat import SoccerStat

from agents.MADDPG_v2.maddpg import MADDPG
from agents.common.training_opponent import StationaryOpponent, RandomSwitchOpponent, RLBasedOpponent
from agents.utils.adjust_state import normalize, state_each

if len(sys.argv) != 4:
    print('Usage: python play_with_model.py <model-path> <episode> <deterministic>')
    sys.exit()
else:
    moving_avg_file = sys.argv[1]
    model_path = sys.argv[2]
    episode = sys.argv[3]
    isDeterministic = bool(int(sys.argv[4]))

# set environment
env = SoccerEnv(width=5, height=5, goal_size=3)

# set agents
agentL = MADDPG(act_dim=env.act_dim, env_dim=env.env_dim)
agent_type = model_path.split('/')[1]
actor = f'agentL_{agent_type}_{episode}_LR_1e-05_actor.h5'
critic = f'agentL_{agent_type}_{episode}_LR_1e-05_critic.h5'
agentL.load_weights(path_actor=model_path+actor, path_critic=model_path+critic)

# parameters
EPISODES = 10

for i in range(EPISODES):
    state = env.reset()
    stat.set_initial_ball(state[4])

    stateL, stateR = state_each(normalize(env, state)) # for MADDPG agent

    rewardL = 0
    rewardR = 0
    done = False
    while not done:
        env.show()
        print()

        # agentL decides its action
        if isDeterministic:
            actionL = np.argmax(agentL.policy_action(stateL))
        else:
            actionL = np.random.choice(env.act_dim, p=agentL.policy_action(stateL))

        # agentR decides its action
        actionR = int(input('Choose an action (0~8): '))

        # log information
        print('Probability for each action:')
        print(agentL.policy_action(stateL))
        print('Critic value:')
        print(agentL.critic.target_predict([
            np.expand_dims(stateL, axis=0),
            np.expand_dims(one_hot(actionL, env.act_dim), axis=0),
            np.expand_dims(one_hot(actionR, env.act_dim), axis=0)
        ]))
        print('actionL:', actionL, 'actionR:', actionR)
        print()

        # perform actions on the environment
        done, reward_l, reward_r, state_, actions = env.step(actionL, actionR)

        agentR.adjust(done, reward_r, i)

        state = state_
        stateL, stateR = state_each(normalize(env, state))

        rewardL += reward_l
        rewardR += reward_r

        if done:
            print(f'Episode {i+1}: {rewardL} {rewardR}')
            print('======================')
            print()
