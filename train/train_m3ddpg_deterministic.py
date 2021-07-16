import sys
import random
import pickle
import numpy as np
from keras.utils import to_categorical as one_hot

from env import SoccerEnv
from soccer_stat import SoccerStat

from agents.utils.adjust_state import normalize, state_each, state_L2R, state_R2L

if len(sys.argv) != 3:
    print('Usage: python train.py <m3dddpg-version> <moving-avg-log-file>')
    sys.exit()
else:
    moving_avg_file = sys.argv[2]

if sys.argv[1] == 'v0':
    from agents.M3DDPG_v0.m3ddpg import M3DDPG
elif sys.argv[1] == 'v1':
    from agents.M3DDPG_v1.m3ddpg import M3DDPG
else:
    print('Invalid M3DDPG version')
    sys.exit()

# set environment
env = SoccerEnv(width=5, height=5, goal_size=3)

# set agents
agentL = M3DDPG(act_dim=env.act_dim, env_dim=env.env_dim)
agentR = M3DDPG(act_dim=env.act_dim, env_dim=env.env_dim)

# parameters
EPISODES = 10000
epsilon = 0.999 # TODO: move epsilon into the MADDPG class

# statistic
stat = SoccerStat()

for i in range(EPISODES):
    state = env.reset()
    stat.set_initial_ball(state[4])

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
        # Predict target q-values using target networks with minimax
        q_values = []
        for state in new_states:
            action = agentL.actor.target_predict(np.expand_dims(state, axis=0))
            min_q_value = float('inf')
            for op_action in range(env.act_dim):
                op_action = one_hot(op_action, num_classes=env.act_dim)
                q_value = agentL.critic.target_predict([
                    np.expand_dims(state, axis=0),
                    action,
                    np.expand_dims(op_action, axis=0)
                ])[0]
                if q_value < min_q_value:
                    min_q_value = q_value
            q_values.append(min_q_value)
        q_values = np.array(q_values)
        # Compute critic target
        critic_target = agentL.bellman(rewards, q_values, dones)
        # Train both networks on sampled batch, update target networks
        agentL.update_models(
            states,
            one_hot(actions, num_classes=env.act_dim),
            one_hot(op_actions, num_classes=env.act_dim),
            critic_target
        )

        # training process of agentR
        # Add outputs to memory buffer
        agentR.memorize(stateR, actionR, actionL, reward_r, done, stateR_)
        # Sample experience from buffer
        states, actions, op_actions, rewards, dones, new_states, _ = agentR.sample_batch(64)
        # Predict target q-values using target networks with minimax
        q_values = []
        for state in new_states:
            action = agentR.actor.target_predict(np.expand_dims(state, axis=0))
            min_q_value = float('inf')
            for op_action in range(env.act_dim):
                op_action = one_hot(op_action, num_classes=env.act_dim)
                q_value = agentR.critic.target_predict([
                    np.expand_dims(state, axis=0),
                    action,
                    np.expand_dims(op_action, axis=0)
                ])[0]
                if q_value < min_q_value:
                    min_q_value = q_value
            q_values.append(min_q_value)
        q_values = np.array(q_values)
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
            print(f'Episode {i+1}: {rewardL} {rewardR}')
            print(f'epsilon: {epsilon}')

            stat.add_stat(rewardL, rewardR)
            print(*stat.get_moving_avg(), file=open(moving_avg_file, 'a'))
            print('======================')
            print()

            epsilon *= 0.999

    # save trained model
    if (i+1) % 500 == 0 or i == 0:
        agentL.save_weights(f'models/m3ddpg_d/agentL_m3ddpg_d_{i+1}')
        agentR.save_weights(f'models/m3ddpg_d/agentR_m3ddpg_d_{i+1}')

# save training stats
with open('stats/m3ddpg_d.pkl', 'wb') as output:
    pickle.dump(stat, output)
