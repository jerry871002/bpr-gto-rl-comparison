import random
import numpy as np

from env import SoccerEnv

from agents.MADDPG import MADDPG
from agents.common.training_opponent import StationaryOpponent, RandomSwitchOpponent, RLBasedOpponent

# set environment
env = SoccerEnv(width=5, height=5, goal_size=3)

# set agents
ME = MADDPG(act_dim=env.act_dim, env_dim=env.env_dim)
OP = StationaryOpponent(env_width=env.width, env_height=env.height, env_goal_size=env.goal_size)

# parameters
EPISODES = 1000
epsilon = 0.999

for i in range(EPISODES):
    _, _, _, state, _ = env.reset()

    done = False
    while not done:
        env.show()

        # agent 1 decides its action
        if random.random() > epsilon:
            actionME = np.argmax(ME.policy_action(state))
        else:
            actionME = random.choices(np.arange(5), ME.policy_action(state))[0]

        # agent 2 decides its action
        actionOP = OP.get_action(state)

        # perform actions on the environment
        done, reward_l, reward_r, state_, actions = env.step(actionME, actionOP)

        # training process of agent 1
        # Add outputs to memory buffer
        ME.memorize(state, actionME, actionOP, reward_l, done, state_)
        # Sample experience from buffer
        states, actions, op_actions, rewards, dones, new_states, _ = ME.sample_batch(64)
        # Predict target q-values using target networks
        op_actions_new = [agent.policyOP(TYPEop, state[0], state[1]) \
                          for state in new_states]
        op_actions_new = one_hot(op_actions_new, num_classes=5)

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
            one_hot(actions, num_classes=5),
            one_hot(op_actions, num_classes=5),
            critic_target
        )

        # training process of agent 2
        OP.adjust()
