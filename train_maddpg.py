import sys
import random
import numpy as np
from keras.utils import to_categorical as one_hot

from env import SoccerEnv

from agents.MADDPG.maddpg import MADDPG
from agents.common.training_opponent import StationaryOpponent, RandomSwitchOpponent, RLBasedOpponent

if len(sys.argv) != 2:
    print('Usage: python train.py <moving-avg-log-file>')
    sys.exit()
else:
    moving_avg_file = sys.argv[1]

def normalize(state):
    x1, y1, x2, y2, ball = state

    w_norm = env.width - 1
    h_norm = env.height - 1

    x1 = x1 / w_norm
    x2 = x2 / w_norm
    y1 = y1 / h_norm
    y2 = y2 / h_norm

    return (x1, y1, x2, y2, ball)

def state_each(state):
    x1, y1, x2, y2, ball = state

    if ball == 0:
        stateL = (x1, y1, x2, y2, 1)
        stateR = (x2, y2, x1, y1, 0)
    elif ball == 1:
        stateL = (x1, y1, x2, y2, 0)
        stateR = (x2, y2, x1, y1, 1)

    return stateL, stateR

def state_L2R(stateL):
    x1, y1, x2, y2, ball = stateL
    ball = int(not ball)
    return (x2, y2, x1, y1, ball)

def state_R2L(stateR):
    x2, y2, x1, y1, ball = stateR
    ball = int(not ball)
    return (x1, y1, x2, y2, ball)

# set environment
env = SoccerEnv(width=5, height=5, goal_size=3)

# set agents
agentL = MADDPG(act_dim=env.act_dim, env_dim=env.env_dim)
agentR = MADDPG(act_dim=env.act_dim, env_dim=env.env_dim)

# parameters
EPISODES = 5000
epsilon = 0.999 # TODO: move epsilon into the MADDPG class

# record training process
rewardL_history = []
rewardR_history = []

for i in range(EPISODES):
    state = env.reset()

    # adjust the state for each agent
    state = normalize(state)
    stateL, stateR = state_each(state)

    rewardL = 0
    rewardR = 0
    done = False
    while not done:
        env.show()
        print()

        # agentL decides its action
        if random.random() > epsilon:
            actionL = np.argmax(agentL.policy_action(stateL))
        else:
            actionL = random.randint(0, env.act_dim-1)

        # agentR decides its action
        if random.random() > epsilon:
            actionR = np.argmax(agentR.policy_action(stateR))
        else:
            actionR = random.randint(0, env.act_dim-1)

        # log information
        print('Probability for each action:')
        print('L:', agentL.policy_action(stateL))
        print('R:', agentR.policy_action(stateR))
        print('Critic value:')
        print('L: ', end='')
        print(agentL.critic.target_predict([
            np.expand_dims(stateL, axis=0),
            np.expand_dims(one_hot(actionL, env.act_dim), axis=0),
            np.expand_dims(one_hot(actionR, env.act_dim), axis=0)
        ]))
        print('R: ', end='')
        print(agentR.critic.target_predict([
            np.expand_dims(stateR, axis=0),
            np.expand_dims(one_hot(actionR, env.act_dim), axis=0),
            np.expand_dims(one_hot(actionL, env.act_dim), axis=0)
        ]))
        print('actionL:', actionL, 'actionR:', actionR)
        print()

        # perform actions on the environment
        done, reward_l, reward_r, state_, actions = env.step(actionL, actionR)

        # adjust the state for each agent
        state_ = normalize(state_)
        stateL_, stateR_ = state_each(state_)

        # training process of agentL
        # Add outputs to memory buffer
        agentL.memorize(stateL, actionL, actionR, reward_l, done, stateL_)
        # Sample experience from buffer
        states, actions, op_actions, rewards, dones, new_states, _ = agentL.sample_batch(64)
        # Predict target q-values using target networks
        op_actions_new = [np.argmax(agentR.policy_action(state_L2R(state))) \
                          for state in new_states]
        op_actions_new = one_hot(op_actions_new, num_classes=env.act_dim)

        q_values = agentL.critic.target_predict([
            new_states,
            agentL.actor.target_predict(new_states),
            op_actions_new
        ])
        # Compute critic target
        critic_target = agentL.bellman(rewards, q_values, dones)
        # Train both networks on sampled batch, update target networks
        agentL.update_models(
            states,
            one_hot(actions, num_classes=env.act_dim),
            one_hot(op_actions, num_classes=env.act_dim),
            critic_target
        )

        # training process of agent 2
        # Add outputs to memory buffer
        agentR.memorize(stateR, actionR, actionL, reward_r, done, stateR_)
        # Sample experience from buffer
        states, actions, op_actions, rewards, dones, new_states, _ = agentR.sample_batch(64)
        # Predict target q-values using target networks
        op_actions_new = [np.argmax(agentL.policy_action(state_R2L(state))) \
                          for state in new_states]
        op_actions_new = one_hot(op_actions_new, num_classes=env.act_dim)

        q_values = agentR.critic.target_predict([
            new_states,
            agentR.actor.target_predict(new_states),
            op_actions_new
        ])
        # Compute critic target
        critic_target = agentR.bellman(rewards, q_values, dones)
        # Train both networks on sampled batch, update target networks
        agentR.update_models(
            states,
            one_hot(actions, num_classes=env.act_dim),
            one_hot(op_actions, num_classes=env.act_dim),
            critic_target
        )

        state = state_
        stateL, stateR = state_each(state)

        rewardL += reward_l
        rewardR += reward_r

        if done:
            rewardL_history.append(rewardL)
            rewardR_history.append(rewardR)
            print(f'Episode {i+1}: {rewardL} {rewardR}')
            print(f'epsilon: {epsilon}')
            print(np.mean(rewardL_history[-100:]),
                  np.mean(rewardR_history[-100:]),
                  file=open(moving_avg_file, 'a'))
            print('======================')
            print()

            epsilon *= 0.999
