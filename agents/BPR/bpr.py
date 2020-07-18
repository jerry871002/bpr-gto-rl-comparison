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


class BPR:
    def __init__(self, env_width, env_height, env_goal_size):
        self.belief_attack = np.ones(3) / 3
        self.belief_defense = np.ones(5) / 5
        self.policy_attack = random.randint(0, 2)
        self.policy_defense = random.randint(0, 4)
        self.env_width = env_width
        self.env_height = env_height
        self.attacking = True
        self.back_to_origin = True
        self.lr_prior = 0.05

    def update_belief(self, state, actionOP, OP_prob_random=0.05):
        MEx, MEy, OPx, OPy, possession = state
        prob_distribution = [1 - OP_prob_random] + [OP_prob_random / 19 * i for i in [4, 3, 2, 1, 2, 3, 4]]
        if possession == 0:  # left ball possession
            # likelihood is the prob distribution of OP type 0,1,2
            likelihood_attack = self.performance_model_attack(MEx, MEy, OPx, OPy, actionOP, prob_distribution)

            self.belief_attack = self.belief_attack * (1 - self.lr_prior) + \
                                 (self.belief_attack * likelihood_attack / np.sum(
                                     self.belief_attack * likelihood_attack)) * self.lr_prior

            self.belief_attack = self.belief_attack ** 0.9 / np.sum(self.belief_attack ** 0.9)

        if possession == 1:  # right ball possession
            # likelihood is the prob distribution of OP type 0,1,2,3,4
            likelihood_defense = self.performance_model_defense(MEx, MEy, OPx, OPy, actionOP, prob_distribution)

            self.belief_defense = self.belief_defense * (1 - self.lr_prior) + \
                                  (self.belief_defense * likelihood_defense / np.sum(
                                      self.belief_defense * likelihood_defense)) * self.lr_prior

            self.belief_defense = self.belief_defense ** 0.9 / np.sum(self.belief_defense ** 0.9)

    def performance_model_attack(self, MEx, MEy, OPx, OPy, actionOP, prob):  # Me attack, OP defense
        l = len(prob)
        performance = np.zeros((4, 3, l))

        performance[0, 0, 0:l] = self.rotate(prob, 1)
        performance[0, 1, 0:l] = self.rotate(prob, 2)
        performance[0, 2, 0:l] = self.rotate(prob, 3)

        performance[1, 0, 0:l] = self.rotate(prob, 3)
        performance[1, 1, 0:l] = self.rotate(prob, 3)
        performance[1, 2, 0:l] = self.rotate(prob, 4)

        performance[2, 0, 0:l] = self.rotate(prob, 0)
        performance[2, 1, 0:l] = self.rotate(prob, 1)
        performance[2, 2, 0:l] = self.rotate(prob, 1)

        performance[3, 0, 0:l] = self.rotate(prob, 0)
        performance[3, 1, 0:l] = self.rotate(prob, 2)
        performance[3, 2, 0:l] = self.rotate(prob, 4)

        if OPy == MEy and OPx == MEx + 2:
            return performance[0, :, actionOP]
        elif OPy == MEy - 1 and OPx == MEx + 1:
            return performance[1, :, actionOP]
        elif OPy == MEy + 1 and OPx == MEx + 1:
            return performance[2, :, actionOP]
        elif OPy == MEy and OPx == MEx + 1:
            return performance[3, :, actionOP]

        return np.ones(3) / 3

    def performance_model_defense(self, MEx, MEy, OPx, OPy, actionOP, prob):  # Me defense, OP attack
        l = len(prob)
        performance = np.zeros((5, 5, l))
        performance[0, 0, 0:l] = self.rotate(prob, 2)
        performance[0, 1, 0:l] = self.rotate(prob, 4)
        performance[0, 2, 0:l] = self.rotate(prob, 4)
        performance[0, 3, 0:l] = self.rotate(prob, 4)
        performance[0, 4, 0:l] = self.rotate(prob, 4)

        performance[1, 0, 0:l] = self.rotate(prob, 0)
        performance[1, 1, 0:l] = self.rotate(prob, 2)
        performance[1, 2, 0:l] = self.rotate(prob, 4)
        performance[1, 3, 0:l] = self.rotate(prob, 4)
        performance[1, 4, 0:l] = self.rotate(prob, 4)

        performance[2, 0, 0:l] = self.rotate(prob, 0)
        performance[2, 1, 0:l] = self.rotate(prob, 0)
        performance[2, 2, 0:l] = self.rotate(prob, 2)
        performance[2, 3, 0:l] = self.rotate(prob, 4)
        performance[2, 4, 0:l] = self.rotate(prob, 4)

        performance[3, 0, 0:l] = self.rotate(prob, 0)
        performance[3, 1, 0:l] = self.rotate(prob, 0)
        performance[3, 2, 0:l] = self.rotate(prob, 0)
        performance[3, 3, 0:l] = self.rotate(prob, 2)
        performance[3, 4, 0:l] = self.rotate(prob, 4)

        performance[4, 0, 0:l] = self.rotate(prob, 0)
        performance[4, 1, 0:l] = self.rotate(prob, 0)
        performance[4, 2, 0:l] = self.rotate(prob, 0)
        performance[4, 3, 0:l] = self.rotate(prob, 0)
        performance[4, 4, 0:l] = self.rotate(prob, 2)

        if OPx != 0:
            if OPy == 0:
                return performance[0, :, actionOP]
            elif OPy == 1:
                return performance[1, :, actionOP]
            elif OPy == 2:
                return performance[2, :, actionOP]
            elif OPy == 3:
                return performance[3, :, actionOP]
            elif OPy == 4:
                return performance[4, :, actionOP]
        return np.ones(5) / 5

    def rotate(self, l, n):
        return l[n:] + l[:n]

    # if ball possession change -> change policy    
    def change_policy(self, MEx, MEy, ball_possession):
        if ball_possession == 0:  # ME possess ball
            self.attacking = True
            self.policy_attack = np.argmin(self.belief_attack)
            if MEy == 0 and self.policy_attack == 0:
                self.policy_attack = 1
            if MEy == self.env_height and self.policy_attack == 3:
                self.policy_attack = 1
        elif ball_possession == 1:  # OP possess ball
            if MEx != 0 or MEy != int(self.env_height / 2):
                self.back_to_origin = False
            self.attacking = False
            self.policy_defense = np.argmax(self.belief_defense)
            print('defense belief = ', self.belief_defense)

    # choose action according to policy
    def choose_action(self, state):
        MEx, MEy, OPx, OPy, possession = state
        # attacking
        if self.attacking:
            # in front of goal
            if MEx == self.env_width - 1 and MEy == self.env_height - 1:
                return TOP_RIGHT
            if MEx == self.env_width - 1 and MEy == 0:
                return BOTTOM_RIGHT
            # close to opponent, show policy
            if (abs(MEx - OPx) + abs(MEy - OPy) <= 2 and MEx < OPx):
                if self.policy_attack == 0:  # attack from top
                    return self.to_target(MEx, MEy, OPx, OPy - 1)
                if self.policy_attack == 1:  # attack from middle
                    return self.to_target(MEx, MEy, OPx, OPy)
                if self.policy_attack == 2:  # attack from bottom
                    return self.to_target(MEx, MEy, OPx, OPy + 1)
            # or (MEx == OPx and MEy == OPy-1) or (MEx == OPx and MEy == OPy+1)

            return RIGHT
        # defending
        elif not self.attacking:
            # go back to origin first
            if MEx == 0 and MEy == int(self.env_height / 2):
                self.back_to_origin = True
            if not self.back_to_origin:
                return self.to_target(MEx, MEy, 0, int(self.env_height / 2))
            # try to block OP
            if self.policy_defense == 0:
                return self.move_to_row(state[1], 0)
            elif self.policy_defense == 1:
                return self.move_to_row(state[1], int(self.env_height / 4))
            elif self.policy_defense == 2:
                return self.move_to_row(state[1], int(self.env_height / 2))
            elif self.policy_defense == 3:
                return self.move_to_row(state[1], int(self.env_height / 4 * 3))
            elif self.policy_defense == 4:
                return self.move_to_row(state[1], self.env_height - 1)

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
