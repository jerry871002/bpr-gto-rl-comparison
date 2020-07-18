"""BPR left agent with 3 attacking policies and 3 defending policies"""

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


class BPR_3_3:
    def __init__(self, env_width, env_height, env_goal_size):
        self.belief_attack = np.ones(3) / 3
        self.belief_defense = np.ones(3) / 3
        self.policy_attack = random.randint(0, 2)
        self.policy_defense = random.randint(0, 2)
        self.env_width = env_width
        self.env_height = env_height

    def update_belief(self, state, actionOP, OP_prob_random=0.05):
        MEx, MEy, OPx, OPy, possession = state
        prob_distribution = [1 - OP_prob_random] + [OP_prob_random / 19 * i for i in [4, 3, 2, 1, 2, 3, 4]]
        if possession == 0:  # left ball possession
            # likelihood is the prob distribution of OP type 0, 1, 2
            likelihood_attack = self.performance_model_attack(MEx, MEy, OPx, OPy, actionOP, prob_distribution)
            self.belief_attack = self.belief_attack * likelihood_attack / np.sum(self.belief_attack * likelihood_attack)
            self.belief_attack = self.belief_attack ** 0.9 / np.sum(self.belief_attack ** 0.9)
            for i in range(len(self.belief_attack)):
                if self.belief_attack[i] < 1e-5:
                    self.belief_attack[i] = 1e-5

        if possession == 1:  # right ball possession
            # likelihood is the prob distribution of OP type 0, 1, 2, 3, 4
            likelihood_defense = self.performance_model_defense(MEx, MEy, OPx, OPy, actionOP, prob_distribution)
            self.belief_defense = self.belief_defense * likelihood_defense / np.sum(
                self.belief_defense * likelihood_defense)
            self.belief_defense = self.belief_defense ** 0.9 / np.sum(self.belief_defense ** 0.9)
            for i in range(len(self.belief_defense)):
                if self.belief_defense[i] < 1e-10:
                    self.belief_defense[i] = 1e-10

    # position  0     |  1     |  2     |  3     |  4     |  5    |  6    |
    #         -------------------------------------------------------------
    #          .....  | .....  | .....  | .....  | .....  | ..... | ..... |
    #          ...x.  | .....  | .....  | ..x..  | .....  | ..... | ..... |
    #          .o...  | .o.x.  | .o...  | .o...  | .ox..  | .o... | ..... |
    #          .....  | .....  | ...x.  | .....  | .....  | ..x.. | ..... |
    #          .....  | .....  | .....  | .....  | .....  | ..... | ..... |

    def performance_model_attack(self, MEx, MEy, OPx, OPy, actionOP, prob):  # Me attack, OP defense
        l = len(prob)
        performance = np.zeros((6, 3, l))
        # position 0, type 0, 1, 2
        performance[0, 0, 0:l] = self.rotate(prob, 2)
        performance[0, 1, 0:l] = self.rotate(prob, 3)
        performance[0, 2, 0:l] = self.rotate(prob, 3)
        # position 1
        performance[1, 0, 0:l] = self.rotate(prob, 1)
        performance[1, 1, 0:l] = self.rotate(prob, 2)
        performance[1, 2, 0:l] = self.rotate(prob, 3)
        # position 2
        performance[2, 0, 0:l] = self.rotate(prob, 1)
        performance[2, 1, 0:l] = self.rotate(prob, 1)
        performance[2, 2, 0:l] = self.rotate(prob, 2)
        # position 3
        performance[3, 0, 0:l] = self.rotate(prob, 3)
        performance[3, 1, 0:l] = self.rotate(prob, 3)
        performance[3, 2, 0:l] = self.rotate(prob, 4)
        # position 4
        performance[4, 0, 0:l] = self.rotate(prob, 0)
        performance[4, 1, 0:l] = self.rotate(prob, 2)
        performance[4, 2, 0:l] = self.rotate(prob, 4)
        # position 5
        performance[5, 0, 0:l] = self.rotate(prob, 0)
        performance[5, 1, 0:l] = self.rotate(prob, 1)
        performance[5, 2, 0:l] = self.rotate(prob, 1)

        if OPx == MEx + 2:
            if OPy == MEy - 1:
                return performance[0, :, actionOP]
            elif OPy == MEy:
                return performance[1, :, actionOP]
            elif OPy == MEy + 1:
                return performance[2, :, actionOP]
        elif OPx == MEx + 1:
            if OPy == MEy - 1:
                return performance[3, :, actionOP]
            elif OPy == MEy:
                return performance[4, :, actionOP]
            elif OPy == MEy + 1:
                return performance[5, :, actionOP]
        return np.ones(3) / 3

    def performance_model_defense(self, MEx, MEy, OPx, OPy, actionOP, prob):  # Me defense, OP attack
        l = len(prob)
        performance = np.zeros((6, 3, l))

        performance[0, 0, 0:l] = self.rotate(prob, 2)
        performance[0, 1, 0:l] = self.rotate(prob, 3)
        performance[0, 2, 0:l] = self.rotate(prob, 3)
        # position 1
        performance[1, 0, 0:l] = self.rotate(prob, 1)
        performance[1, 1, 0:l] = self.rotate(prob, 2)
        performance[1, 2, 0:l] = self.rotate(prob, 3)
        # position 2
        performance[2, 0, 0:l] = self.rotate(prob, 1)
        performance[2, 1, 0:l] = self.rotate(prob, 1)
        performance[2, 2, 0:l] = self.rotate(prob, 2)
        # position 3
        performance[3, 0, 0:l] = self.rotate(prob, 2)
        performance[3, 1, 0:l] = self.rotate(prob, 3)
        performance[3, 2, 0:l] = self.rotate(prob, 3)
        # position 4
        performance[4, 0, 0:l] = self.rotate(prob, 1)
        performance[4, 1, 0:l] = self.rotate(prob, 2)
        performance[4, 2, 0:l] = self.rotate(prob, 3)
        # position 5
        performance[5, 0, 0:l] = self.rotate(prob, 1)
        performance[5, 1, 0:l] = self.rotate(prob, 1)
        performance[5, 2, 0:l] = self.rotate(prob, 2)

        if OPx == MEx + 2:
            if OPy == MEy - 1:
                return performance[0, :, actionOP]
            elif OPy == MEy:
                return performance[1, :, actionOP]
            elif OPy == MEy + 1:
                return performance[2, :, actionOP]
        elif OPx == MEx + 1:
            if OPy == MEy - 1:
                return performance[3, :, actionOP]
            elif OPy == MEy:
                return performance[4, :, actionOP]
            elif OPy == MEy + 1:
                return performance[5, :, actionOP]
        return np.ones(3) / 3

    def rotate(self, l, n):
        return l[n:] + l[:n]

    # if ball possession change -> change policy    
    def change_policy(self, MEx, MEy, ball_possession):
        if ball_possession == 0:  # left possess ball
            self.policy_attack = np.argmin(self.belief_attack)
            if MEy == 0 and self.policy_attack == 0:
                self.policy_attack = 1
            if MEy == self.env_height and self.policy_attack == 3:
                self.policy_attack = 1
            print('agent_l attack belief = ', self.belief_attack)
            print('agent_l attack policy = ', self.policy_attack)
        elif ball_possession == 1:  # right possess ball
            self.policy_defense = np.argmax(self.belief_defense)
            print('agent_l defense belief = ', self.belief_defense)
            print('agent_l defense policy = ', self.policy_defense)

    # choose action according to policy
    def choose_action(self, state):
        MEx, MEy, OPx, OPy, possession = state
        # attacking
        if possession == 0:  # left posses ball
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
            return RIGHT
        # defending
        else:  # right possess ball
            # too far from the opponent
            if abs(MEx - OPx) + abs(MEy - OPy) > 2:
                return self.to_target(MEx, MEy, OPx + 2, OPy)
            # close to opponent, show policy
            else:
                if self.policy_defense == 0:
                    target = (OPx - 1, OPy - 1)
                    if (MEx, MEy) != target:
                        return self.to_target(MEx, MEy, *target)
                    else:
                        return self.to_target(MEx, MEy, OPx, OPy)
                elif self.policy_defense == 1:
                    target = (OPx, OPy)
                    if (MEx, MEy) != target:
                        return self.to_target(MEx, MEy, *target)
                    else:
                        return self.to_target(MEx, MEy, OPx, OPy)
                elif self.policy_defense == 2:
                    target = (OPx - 1, OPy + 1)
                    if (MEx, MEy) != target:
                        return self.to_target(MEx, MEy, *target)
                    else:
                        return self.to_target(MEx, MEy, OPx, OPy)

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
