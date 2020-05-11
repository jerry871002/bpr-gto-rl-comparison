import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.initializers import RandomUniform
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, Reshape, GaussianNoise, Flatten, Lambda, concatenate

""" Modified from code by @germain-hug: https://github.com/germain-hug/Deep-RL-Keras/blob/master/DDPG/actor.py
"""

class Actor:
    def __init__(self, inp_dim, out_dim, lr, tau):
        self.env_dim = inp_dim
        self.act_dim = out_dim
        self.tau = tau
        self.lr = lr

        self.model, self.training_model = self.network()
        self.target_model, _ = self.network()
        self.target_model.set_weights(self.model.get_weights())

        self.model.compile(Adam(lr), loss=self.actor_loss)
        self.target_model.compile(Adam(lr), loss=self.actor_loss)
        self.training_model.compile(Adam(lr), loss=self.actor_loss)

    def network(self):
        """ Actor network to predict probability of each action
        """
        input = Input((self.env_dim,))
        delta = Input((1,))

        # slice input layer
        pos_me = Lambda(lambda x: x[:, 0:2])(input)
        pos_op = Lambda(lambda x: x[:, 2:4])(input)
        ball = Lambda(lambda x: x[:, 4:])(input)

        # combine ball with possition
        pos_me = concatenate([pos_me, ball])
        pos_op = concatenate([pos_op, ball])

        # get features from each input
        pos_me = Dense(32, activation='relu')(pos_me)
        pos_op = Dense(32, activation='relu')(pos_op)

        x = concatenate([pos_me, pos_op])
        x = Dense(64, activation='relu')(x)
        output = Dense(self.act_dim, activation='softmax', kernel_initializer=RandomUniform())(x)

        return Model(input, output), Model(input=[input, delta], output=output)

    def actor_loss(self, y_true, y_pred):
        out = K.clip(y_pred, 1e-8, 1-1e-8)
        log_lik = y_true * K.log(out)

        return K.sum(-log_lik * self.training_model.input[1])

    def predict(self, state):
        """ Action prediction
        """
        return self.model.predict(np.expand_dims(state, axis=0))

    def target_predict(self, input):
        """ Action prediction (target network)
        """
        return self.target_model.predict(input)

    def transfer_weights(self):
        """ Transfer model weights to target model with a factor of Tau
        """
        W, target_W = self.model.get_weights(), self.target_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau)* target_W[i]
        self.target_model.set_weights(target_W)

    def train(self, states, actions, q_values, critic_target):
        td_error = critic_target - q_values
        self.training_model.train_on_batch([states, td_error], actions)

    def save(self, path):
        self.model.save_weights(path + '_actor.h5')

    def load_weights(self, path):
        self.model.load_weights(path)
