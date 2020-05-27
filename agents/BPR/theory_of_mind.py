#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 19:32:22 2020

@author: yenyunhuang
"""
import numpy as np
import random
import math

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
        self.zero_belief_attack = np.ones(3)/3
        self.zero_belief_defense = np.ones(5)/5
        self.first_belief_attack = np.ones(3)/3
        self.first_belief_defense = np.ones(5)/5
        self.policy_attack = random.randint(0, 2)
        self.policy_defense = random.randint(0, 4)
        self.env_width = env_width
        self.env_height = env_height
        self.attacking = True
        self.back_to_origin = True
        self.confidence = 0.3
        self.adjustment_rate = 0.7
        self.winrate_threshold = 0.7
        self.indicator_value = 0
        
    def update_confidence(self, current_win_rate, past_win_rate):
        if current_win_rate >= past_win_rate:
            return ((1-self.adjustment_rate) * self.confidence + self.adjustment_rate) * self.indicator_value
        elif self.winrate_threshold < current_win_rate < past_win_rate:
            temp = math.log(current_win_rate, 10)/math.log(current_win_rate-self.winrate_threshold, 10)
            return temp * self.confidence * self.indicator_value
        else:
            return self.adjustment_rate * self.indicator_value
        
    def indicator(self, win_rate):
        if win_rate <= self.winrate_threshold and self.indicator_value == 0:
            self.indicator_value = 1
        elif win_rate <= self.winrate_threshold and self.indicator_value == 1:
            self.indicator_value = 0
                
    def update_zero_order_belief(self, state, actionRight, prob_random=0.05):
        leftx, lefty, rightx, righty, possession = state
        prob_distribution = [1-prob_random] + [prob_random/19*i for i in [4, 3, 2, 1, 2, 3, 4]]
        if possession == 0: #left ball possession
            #likelihood is the prob distribution of OP type 0,1,2
            likelihood_attack = self.my_performance_model_attack(leftx, lefty, rightx, righty, actionRight, prob_distribution)
            self.zero_belief_attack = self.zero_belief_attack*likelihood_attack/np.sum(self.zero_belief_attack*likelihood_attack)
            self.zero_belief_attack = self.zero_belief_attack**0.9/np.sum(self.zero_belief_attack**0.9)
            # for i in range(len(self.belief_attack)):
            #     if self.belief_attack[i] < 1e-5:
            #         self.belief_attack[i] = 1e-5
            
        else: #right ball possession
            #likelihood is the prob distribution of OP type 0,1,2,3,4
            likelihood_defense = self.my_performance_model_defense(leftx, lefty, rightx, righty, actionRight, prob_distribution)
            self.zero_belief_defense = self.zero_belief_defense*likelihood_defense/np.sum(self.zero_belief_defense*likelihood_defense)
            self.zero_belief_defense = self.zero_belief_defense**0.9/np.sum(self.zero_belief_defense**0.9)
            # for i in range(len(self.belief_defense)):
            #     if self.belief_defense[i] < 1e-10:
            #         self.belief_defense[i] = 1e-10
        
    # still need to think about the meaning of updating first order belief !!!    
    def update_first_order_belief(self, state, actionLeft, prob_random=0.05):
        leftx, lefty, rightx, righty, possession = state
        prob_distribution = [1-prob_random] + [prob_random/19*i for i in [4, 3, 2, 1, 2, 3, 4]]
        if possession == 0: #left possess ball # op is defending
            likelihood_attack = self.op_performance_model_defense(rightx, righty, leftx, lefty, actionLeft, prob_distribution)
            self.first_belief_attack = self.first_belief_attack*likelihood_attack/np.sum(self.first_belief_attack*likelihood_attack)
            self.first_belief_attack = self.first_belief_attack**0.9/np.sum(self.first_belief_attack**0.9)
            
        else: # right possess ball # op is attacking
            likelihood_defense = self.op_performance_model_attack(rightx, righty, leftx, lefty, actionLeft, prob_distribution)
            self.first_belief_defense = self.first_belief_defense*likelihood_defense/np.sum(self.first_belief_defense*likelihood_defense)
            self.first_belief_defense = self.first_belief_defense**0.9/np.sum(self.first_belief_defense**0.9)
            
    def rotate(self, l, n):
        return l[n:] + l[:n]
    
    def op_performance_model_attack(self, rightx, righty, leftx, lefty, actionLeft, prob): #Me defense, OP attack
        l = len(prob)
        performance = np.zeros((3, 5, l))
        
        performance[0, 0, 0:l] = self.rotate(prob, 0)
        performance[0, 1, 0:l] = self.rotate(prob, -2)
        performance[0, 2, 0:l] = self.rotate(prob, 4)
        performance[0, 3, 0:l] = self.rotate(prob, 4)
        performance[0, 4, 0:l] = self.rotate(prob, 4)
        
        performance[1, 0, 0:l] = self.rotate(prob, 0)
        performance[1, 1, 0:l] = self.rotate(prob, 0)
        performance[1, 2, 0:l] = self.rotate(prob, -2)
        performance[1, 3, 0:l] = self.rotate(prob, 4)
        performance[1, 4, 0:l] = self.rotate(prob, 4)
        
        performance[2, 0, 0:l] = self.rotate(prob, 0)
        performance[2, 1, 0:l] = self.rotate(prob, 0)
        performance[2, 2, 0:l] = self.rotate(prob, 0)
        performance[2, 3, 0:l] = self.rotate(prob, -2)
        performance[2, 4, 0:l] = self.rotate(prob, 4)
        # print(performance)
        # performance = np.negative(performance)
        if leftx != 0 or leftx != 1:
            if lefty==int(self.env_height/4): return performance[0,:,actionLeft]
            elif lefty==int(self.env_height/2): return performance[1,:,actionLeft]
            elif lefty==int(self.env_height/4*3): return performance[2,:,actionLeft]
        return np.ones(5)/5
    
    def op_performance_model_defense(self, MEx, MEy, OPx, OPy, actionOP, prob): #Me attack, OP defense
        l = len(prob)
        performance = np.zeros((3, 3, l))
        
        performance[0, 0, 0:l] = self.rotate(prob, -2)
        performance[0, 1, 0:l] = self.rotate(prob, -3)
        performance[0, 2, 0:l] = self.rotate(prob, -3)
      
        performance[1, 0, 0:l] = self.rotate(prob, -1)
        performance[1, 1, 0:l] = self.rotate(prob, -2)
        performance[1, 2, 0:l] = self.rotate(prob, -3)
        
        performance[2, 0, 0:l] = self.rotate(prob, -1)
        performance[2, 1, 0:l] = self.rotate(prob, -1)
        performance[2, 2, 0:l] = self.rotate(prob, -2)
        # print(performance)
        # performance = np.negative(performance)
        if 0 < MEx - OPx <= 2:
            if OPy<MEy: return performance[0,:,actionOP]
            elif OPy==MEy: return performance[1,:,actionOP]
            elif OPy>MEy: return performance[2,:,actionOP]
            # return 0
        else:
            return np.ones(3)/3
    
    def my_performance_model_attack(self, leftx, lefty, rightx, righty, actionRight, prob): #Me attack, OP defense
        l = len(prob)
        performance = np.zeros((3, 3, l))
        
        performance[0, 0, 0:l] = self.rotate(prob, 2)
        performance[0, 1, 0:l] = self.rotate(prob, 3)
        performance[0, 2, 0:l] = self.rotate(prob, 4)
        
        performance[1, 0, 0:l] = self.rotate(prob, 0)
        performance[1, 1, 0:l] = self.rotate(prob, 2)
        performance[1, 2, 0:l] = self.rotate(prob, 4)
        
        performance[2, 0, 0:l] = self.rotate(prob, 0)
        performance[2, 1, 0:l] = self.rotate(prob, 1)
        performance[2, 2, 0:l] = self.rotate(prob, 2)
        # print(performance)
        if rightx - leftx <= 2:
            if righty<lefty: return performance[0,:,actionRight]
            elif righty==lefty: return performance[1,:,actionRight]
            elif righty>lefty: return performance[2,:,actionRight]
            # return 0
        else:
            return np.ones(3)/3
        
    def my_performance_model_defense(self, leftx, lefty, rightx, righty, actionRight, prob): #Me defense, OP attack
        l = len(prob)
        performance = np.zeros((3, 5, l))
        
        performance[0, 0, 0:l] = self.rotate(prob, 0)
        performance[0, 1, 0:l] = self.rotate(prob, 2)
        performance[0, 2, 0:l] = self.rotate(prob, 4)
        performance[0, 3, 0:l] = self.rotate(prob, 4)
        performance[0, 4, 0:l] = self.rotate(prob, 4)
        
        performance[1, 0, 0:l] = self.rotate(prob, 0)
        performance[1, 1, 0:l] = self.rotate(prob, 0)
        performance[1, 2, 0:l] = self.rotate(prob, 2)
        performance[1, 3, 0:l] = self.rotate(prob, 4)
        performance[1, 4, 0:l] = self.rotate(prob, 4)
        
        performance[2, 0, 0:l] = self.rotate(prob, 0)
        performance[2, 1, 0:l] = self.rotate(prob, 0)
        performance[2, 2, 0:l] = self.rotate(prob, 0)
        performance[2, 3, 0:l] = self.rotate(prob, 2)
        performance[2, 4, 0:l] = self.rotate(prob, 4)
        # print(performance)
        if rightx != 0 or rightx != 1:
            if righty==int(self.env_height/4): return performance[0,:,actionRight]
            elif righty==int(self.env_height/2): return performance[1,:,actionRight]
            elif righty==int(self.env_height/4*3): return performance[2,:,actionRight]
        return np.ones(5)/5
    
    # if ball possession change -> change policy    
    def change_policy(self, leftx, lefty, ball_possession):
        if ball_possession == 0: #ME possess ball
            first_order_policy = np.argmin(self.first_belief_attack)
            integrate_attack = np.empty(3)
            for i in range(3):
                if first_order_policy == i:
                    integrate_attack[i] = (1-self.confidence) * self.zero_belief_attack[i] + self.confidence
                else:
                    integrate_attack[i] = (1-self.confidence) * self.zero_belief_attack[i]
            
            
            
            self.attacking = True
            if lefty == 0 and self.policy_attack == 0:
                self.policy_attack = 1
            if lefty == self.env_height and self.policy_attack == 3:
                self.policy_attack = 1
            # print('attack belief = ', self.belief_attack)
            # print('attack policy = ', self.policy_attack)
        elif ball_possession == 1: #OP possess ball
            if lefty != 0 or lefty != int(self.env_height/2):
                self.back_to_origin = False
            self.attacking = False
            self.policy_defense = np.argmax(self.belief_defense)
            print('defense belief = ', self.belief_defense)
            # print('defense policy = ', self.policy_defense)
    
    #choose action according to policy
    def choose_action(self, state):
        leftx, lefty, rightx, righty, possession = state
        #attacking
        if self.attacking:
            # in front of goal
            if leftx == self.env_width-1 and lefty == self.env_height-1:
                return TOP_RIGHT
            if leftx == self.env_width-1 and lefty == 0:
                return BOTTOM_RIGHT
            # close to opponent, show policy
            if (abs(leftx - rightx) + abs(lefty - righty) <= 2 and leftx < rightx):
                if self.policy_attack == 0: # attack from top
                    return self.to_target(leftx, lefty, rightx, righty-1)
                if self.policy_attack == 1: # attack from middle
                    return self.to_target(leftx, lefty, rightx, righty)
                if self.policy_attack == 2: # attack from bottom
                    return self.to_target(leftx, lefty, rightx, righty+1)
            # or (MEx == OPx and MEy == OPy-1) or (MEx == OPx and MEy == OPy+1)
            
            return RIGHT
        # defending
        elif not self.attacking:
            # print('back to origin: ', self.back_to_origin)
            #go back to origin first
            if leftx == 0 and lefty == int(self.env_height/2):
                self.back_to_origin = True
            if not self.back_to_origin:
                return self.to_target(leftx, lefty, 0, int(self.env_height/2))
            # try to block OP
            if self.policy_defense == 0:
                return self.move_to_row(state[1], 0)
            elif self.policy_defense == 1:
                return self.move_to_row(state[1], int(self.env_height/4))
            elif self.policy_defense == 2:
                return self.move_to_row(state[1], int(self.env_height/2))
            elif self.policy_defense == 3:
                return self.move_to_row(state[1], int(self.env_height/4*3))
            elif self.policy_defense == 4:
                return self.move_to_row(state[1], self.env_height-1)
    
    def to_target(self, MEx, MEy, TARGETx, TARGETy):
        if MEx > TARGETx and MEy > TARGETy:
            return TOP_LEFT
        if MEx > TARGETx and MEy < TARGETy:
            return BOTTOM_LEFT
        if MEx > TARGETx and MEy == TARGETy:
            return LEFT
        if MEx < TARGETx and MEy > TARGETy:
            return TOP_RIGHT
        if MEx < TARGETx and MEy < TARGETy:
            return BOTTOM_RIGHT
        if MEx < TARGETx and MEy == TARGETy:
            return RIGHT
        if MEx == TARGETx and MEy > TARGETy:
            return TOP
        if MEx == TARGETx and MEy < TARGETy:
            return BOTTOM
        
    
    def move_to_row(self, y, target_row):
        if y < target_row:
            return BOTTOM
        elif y > target_row:
            return TOP
        else:
            return RIGHT