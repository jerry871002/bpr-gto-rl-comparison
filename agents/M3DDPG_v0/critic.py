import numpy as np
import keras.backend as K

from keras.initializers import RandomUniform
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, concatenate, Reshape, Lambda, Flatten

""" Modified from code by @germain-hug: https://github.com/germain-hug/Deep-RL-Keras/blob/master/DDPG/critic.py
"""

class Critic:
    """ Critic for the MADDPG Algorithm, Q-Value function approximator
    """

    def __init__(self, inp_dim, out_dim, lr, tau):
        # Dimensions and Hyperparams
        self.env_dim = inp_dim
        self.act_dim = out_dim
        self.tau, self.lr = tau, lr
        # Build models and target models
        self.model = self.network()
        self.target_model = self.network()
        self.target_model.set_weights(self.model.get_weights())

        self.model.compile(Adam(self.lr), 'mse')
        self.target_model.compile(Adam(self.lr), 'mse')
        # Function to compute Q-value gradients (Actor Optimization)
        self.action_grads = K.function([self.model.input[i] for i in range(3)],
                                       K.gradients(self.model.output,
                                                   [self.model.input[1], self.model.input[2]]))

    def network(self):
        """ Assemble Critic network to predict q-values
        """
        state = Input((self.env_dim,))
        action = Input((self.act_dim,))
        op_action = Input((self.act_dim,))

        # slice state
        pos_me = Lambda(lambda x: x[:, 0:2])(state)
        pos_op = Lambda(lambda x: x[:, 2:4])(state)
        ball = Lambda(lambda x: x[:, 4:])(state)

        # combine ball with possition
        pos_me = concatenate([pos_me, ball])
        pos_op = concatenate([pos_op, ball])

        # get features from each input
        action_ = Dense(16, activation='relu')(action)
        op_action_ = Dense(16, activation='relu')(op_action)
        pos_me = Dense(16, activation='relu')(pos_me)
        pos_op = Dense(16, activation='relu')(pos_op)

        x = concatenate([action_, op_action_, pos_me, pos_op])
        x = Dense(64, activation='relu')(x)
        output = Dense(1, activation='linear', kernel_initializer=RandomUniform())(x)

        return Model([state, action, op_action], output)

    def gradients(self, states, actions, op_actions):
        """ Compute Q-value gradients w.r.t. states and policy-actions
        """
        return self.action_grads([states, actions, op_actions])

    def target_predict(self, input):
        """ Predict Q-Values using the target network
        """
        return self.target_model.predict(input)

    def train_on_batch(self, states, actions, op_actions, critic_target):
        """ Train the critic network on batch of sampled experience
        """
        return self.model.train_on_batch([states, actions, op_actions], critic_target)

    def transfer_weights(self):
        """ Transfer model weights to target model with a factor of Tau
        """
        W, target_W = self.model.get_weights(), self.target_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau)* target_W[i]
        self.target_model.set_weights(target_W)

    def save(self, path):
        self.model.save_weights(path + '_critic.h5')

    def load_weights(self, path):
        self.model.load_weights(path)
