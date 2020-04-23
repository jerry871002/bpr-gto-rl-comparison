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
    x, y, x_op, y_op, ball = state
    w_norm = (env.width - 1) / 2
    h_norm = (env.height - 1) / 2
    x = (x - w_norm) / w_norm
    x_op = (x_op - w_norm) / w_norm
    y = (y - h_norm) / h_norm
    y_op = (y_op - h_norm) / h_norm
    ball = 1 if ball else -1
    return (x, y, x_op, y_op, ball)

def resume(state):
    x, y, x_op, y_op, ball = state
    w_norm = (env.width - 1) / 2
    h_norm = (env.height - 1) / 2
    x = int(x * w_norm + w_norm)
    x_op = int(x_op * w_norm + w_norm)
    y = int(y * h_norm + h_norm)
    y_op = int(y_op * h_norm + h_norm)
    ball = 1 if ball == 1 else 0
    return (x, y, x_op, y_op, ball)

# set environment
env = SoccerEnv(width=5, height=5, goal_size=3)

# set agents
ME = MADDPG(act_dim=env.act_dim, env_dim=env.env_dim)
OP = StationaryOpponent(env_width=env.width, env_height=env.height, env_goal_size=env.goal_size)

# parameters
EPISODES = 5000
epsilon = 0.999 # TODO: move epsilon into the MADDPG class

# record training process
reward_history = []

for i in range(EPISODES):
    state = env.reset()

    reward = 0
    done = False
    while not done:
        env.show()
        print()

        # agent 1 decides its action
        if random.random() > epsilon:
            actionME = np.argmax(ME.policy_action(state))
            # actionME = random.choices(np.arange(env.act_dim), ME.policy_action(normalize(state)))[0]
        else:
            # actionME = random.choices(np.arange(env.act_dim), ME.policy_action(state))[0]
            actionME = random.randint(0, env.act_dim-1)

        # agent 2 decides its action
        actionOP = OP.get_action(state)

        # log information
        print('Probability for each action:')
        print(ME.policy_action(normalize(state)))
        print('Critic value:')
        print(ME.critic.target_predict([
            np.expand_dims(normalize(state), axis=0),
            np.expand_dims(one_hot(actionME, env.act_dim), axis=0),
            np.expand_dims(one_hot(actionOP, env.act_dim), axis=0)
        ]))
        print('actionME:', actionME, 'actionOP:', actionOP)
        print()

        # perform actions on the environment
        done, reward_l, reward_r, state_, actions = env.step(actionME, actionOP)

        # training process of agent 1
        # Add outputs to memory buffer
        ME.memorize(normalize(state), actionME, actionOP, reward_l, done, normalize(state_))
        # Sample experience from buffer
        states, actions, op_actions, rewards, dones, new_states, _ = ME.sample_batch(64)
        # Predict target q-values using target networks
        op_actions_new = [OP.get_action(resume(state)) \
                          for state in new_states]
        op_actions_new = one_hot(op_actions_new, num_classes=env.act_dim)

        q_values = ME.critic.target_predict([
            new_states,
            ME.actor.target_predict(new_states),
            op_actions_new
        ])
        # Compute critic target
        critic_target = ME.bellman(rewards, q_values, dones)
        # Train both networks on sampled batch, update target networks
        ME.update_models(
            states,
            one_hot(actions, num_classes=env.act_dim),
            one_hot(op_actions, num_classes=env.act_dim),
            critic_target
        )

        # training process of agent 2
        OP.adjust(done, reward_r, i)

        state = state_
        reward += reward_l

        if done:
            reward_history.append(reward)
            print(f'Episode {i+1}: {reward}')
            print(f'epsilon: {epsilon}')
            print(np.mean(reward_history[-100:]), file=open(moving_avg_file, 'a'))
            print('======================')
            print()

            epsilon *= 0.999
