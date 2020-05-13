#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 19:32:22 2020

@author: yenyunhuang
"""
import numpy as np
import random

TOP = 0
TOP_RIGHT = 1
RIGHT = 2
BOTTOM_RIGHT = 3
BOTTOM = 4
BOTTOM_LEFT = 5
LEFT = 6
TOP_LEFT = 7

class TheoryOfMind:
    def __init__(self, env_width, env_height, env_goal_size):
        self.env_width = env_width
        self.env_height = env_height
        self.env_goal_size = env_goal_size

        self.policy_attack = random.randint(0, 2)
        self.policy_defense = random.randint(0, 4)
        self.belief_attack = np.ones(3)/3
        self.belief_defense = np.ones(5)/5