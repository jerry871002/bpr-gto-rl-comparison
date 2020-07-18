import sys
import numpy as np

from .actor import Actor
from .critic import Critic
from ..utils.memory_buffer import MemoryBuffer

""" Modified from code by @germain-hug: https://github.com/germain-hug/Deep-RL-Keras/blob/master/DDPG/ddpg.py
"""

class M3DDPG:
    def __init__(self, act_dim, env_dim, buffer_size=20000, gamma=0.99, lr=0.00001, tau=0.1):
        """ Initialization
        """
        # Environment and training parameters
        self.act_dim = act_dim
        self.env_dim = env_dim
        self.gamma = gamma
        self.lr = lr
        # Create actor and critic networks
        self.actor = Actor(self.env_dim, self.act_dim, 0.1 * lr, tau)
        self.critic = Critic(self.env_dim, self.act_dim, lr, tau)
        self.buffer = MemoryBuffer(buffer_size)

    def policy_action(self, s):
        """ Use the actor to predict value
        """
        return self.actor.predict(s)[0]

    def bellman(self, rewards, q_values, dones):
        """ Use the Bellman Equation to compute the critic target
        """
        critic_target = np.asarray(q_values)
        for i in range(q_values.shape[0]):
            if dones[i]:
                critic_target[i] = rewards[i]
            else:
                critic_target[i] = rewards[i] + self.gamma * q_values[i]
        return critic_target

    def memorize(self, state, action, op_action, reward, done, new_state):
        """ Store experience in memory buffer
        """
        self.buffer.memorize(state, action, op_action, reward, done, new_state)

    def sample_batch(self, batch_size):
        return self.buffer.sample_batch(batch_size)

    def update_models(self, states, actions, op_actions, critic_target):
        """ Update actor and critic networks from sampled experience
        """
        # Train critic
        self.critic.train_on_batch(states, actions, op_actions, critic_target)
        # Q-Value under Current Policy
        q_values = self.critic.target_predict([
            states,
            self.actor.target_predict(states),
            op_actions
        ])
        # Train actor
        self.actor.train(states, actions, q_values, critic_target)
        # Transfer weights to target networks at rate Tau
        self.actor.transfer_weights()
        self.critic.transfer_weights()

    def save_weights(self, path):
        path += '_LR_{}'.format(self.lr)
        self.actor.save(path)
        self.critic.save(path)

    def load_weights(self, path_actor, path_critic):
        self.critic.load_weights(path_critic)
        self.actor.load_weights(path_actor)
