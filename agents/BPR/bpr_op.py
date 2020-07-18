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


class BPR_OP:
    def __init__(self, env_width, env_height, env_goal_size):
        self.belief_attack = np.ones(5) / 5
        self.belief_defense = np.ones(3) / 3
        self.policy_attack = random.randint(0, 4)
        self.policy_defense = random.randint(0, 2)
        self.env_width = env_width
        self.env_height = env_height
        self.env_goal_size = env_goal_size
        self.attacking = True
        self.back_to_origin = True

    def update_belief(self, state, actionOP, OP_prob_random=0.05):
        prob_distribution = [1 - OP_prob_random] + [OP_prob_random / 19 * i for i in [4, 3, 2, 1, 2, 3, 4]]
        if state[4] == 0:  # left ball possession
            # likelihood is the prob distribution of OP type 0,1,2
            likelihood_defense = self.performance_model_defense(state[2], state[3], state[0], state[1], actionOP,
                                                                prob_distribution)
            self.belief_defense = self.belief_defense * likelihood_defense / np.sum(
                self.belief_defense * likelihood_defense)
            for i in range(len(self.belief_defense)):
                if self.belief_defense[i] < 1e-5:
                    self.belief_defense[i] = 1e-5

        if state[4] == 1:  # right ball possession
            # likelihood is the prob distribution of OP type 0,1,2,3,4
            likelihood_attack = self.performance_model_attack(state[2], state[3], state[0], state[1], actionOP,
                                                              prob_distribution)
            self.belief_attack = self.belief_attack * likelihood_attack / np.sum(self.belief_attack * likelihood_attack)
            for i in range(len(self.belief_attack)):
                if self.belief_attack[i] < 1e-10:
                    self.belief_attack[i] = 1e-10

    def performance_model_defense(self, MEx, MEy, OPx, OPy, actionOP, prob):  # Me attack, OP defense
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

        if 0 < MEx - OPx <= 2:
            if OPy < MEy:
                return performance[0, :, actionOP]
            elif OPy == MEy:
                return performance[1, :, actionOP]
            elif OPy > MEy:
                return performance[2, :, actionOP]
        else:
            return np.ones(3) / 3

    def performance_model_attack(self, MEx, MEy, OPx, OPy, actionOP, prob):  # Me defense, OP attack
        l = len(prob)
        performance = np.zeros((5, 5, l))

        performance[0, 0, 0:l] = self.rotate(prob, -2)
        performance[0, 1, 0:l] = self.rotate(prob, 4)
        performance[0, 2, 0:l] = self.rotate(prob, 4)
        performance[0, 3, 0:l] = self.rotate(prob, 4)
        performance[0, 4, 0:l] = self.rotate(prob, 4)

        performance[1, 0, 0:l] = self.rotate(prob, 0)
        performance[1, 1, 0:l] = self.rotate(prob, -2)
        performance[1, 2, 0:l] = self.rotate(prob, 4)
        performance[1, 3, 0:l] = self.rotate(prob, 4)
        performance[1, 4, 0:l] = self.rotate(prob, 4)

        performance[2, 0, 0:l] = self.rotate(prob, 0)
        performance[2, 1, 0:l] = self.rotate(prob, 0)
        performance[2, 2, 0:l] = self.rotate(prob, -2)
        performance[2, 3, 0:l] = self.rotate(prob, 4)
        performance[2, 4, 0:l] = self.rotate(prob, 4)

        performance[3, 0, 0:l] = self.rotate(prob, 0)
        performance[3, 1, 0:l] = self.rotate(prob, 0)
        performance[3, 2, 0:l] = self.rotate(prob, 0)
        performance[3, 3, 0:l] = self.rotate(prob, -2)
        performance[3, 4, 0:l] = self.rotate(prob, 4)

        performance[4, 0, 0:l] = self.rotate(prob, 0)
        performance[4, 1, 0:l] = self.rotate(prob, 0)
        performance[4, 2, 0:l] = self.rotate(prob, 0)
        performance[4, 3, 0:l] = self.rotate(prob, 0)
        performance[4, 4, 0:l] = self.rotate(prob, -2)

        if OPx != self.env_width:
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
        if ball_possession == 0:  # agent left possess ball
            self.attacking = False
            self.policy_defense = np.argmax(self.belief_defense)
            if MEy == 0 and self.policy_defense == 0:
                self.policy_defense = 1
            if MEy == self.env_height and self.policy_defense == 3:
                self.policy_defense = 1
        elif ball_possession == 1:  # agent right possess ball
            self.attacking = True
            self.policy_attack = np.argmin(self.belief_attack)

    # choose action according to policy
    def choose_action(self, state):
        OPx, OPy, MEx, MEy, possession = state
        # attacking
        if self.attacking:
            # at the start column and in the middle columns
            if 0 < MEx <= self.env_width - 1:
                if self.policy_attack == 0:
                    return self.move_to_row(MEy, 0)
                elif self.policy_attack == 1:
                    return self.move_to_row(MEy, int(self.env_height / 4))
                elif self.policy_attack == 2:
                    return self.move_to_row(MEy, int(self.env_height / 2))
                elif self.policy_attack == 3:
                    return self.move_to_row(MEy, int(self.env_height / 4 * 3))
                elif self.policy_attack == 4:
                    return self.move_to_row(MEy, self.env_height - 1)
            # at the end column
            elif MEx == 0:
                if self.policy_attack == 0:
                    return self.move_to_row(MEy, (self.env_height - self.env_goal_size) / 2)
                elif self.policy_attack == 1:
                    return self.move_to_row(MEy, int((2 * self.env_height - self.env_goal_size) / 4))
                elif self.policy_attack == 2:
                    return self.move_to_row(MEy, int(self.env_height / 2))
                elif self.policy_attack == 3:
                    return self.move_to_row(MEy, int((2 * self.env_height + self.env_goal_size) / 4))
                elif self.policy_attack == 4:
                    return self.move_to_row(MEy, (self.env_height + self.env_goal_size) / 2 - 1)
        # defending
        else:
            # too far from the opponent
            if abs(MEx - OPx) + abs(MEy - OPy) > 2:
                return self.move_to_location(MEx, MEy, OPx + 2, OPy)
            # close to opponent, show policy
            else:
                if self.policy_defense == 0:
                    target = (OPx + 1, OPy - 1)
                    if (MEx, MEy) != target:
                        return self.move_to_location(MEx, MEy, *target)
                    else:
                        return self.move_to_location(MEx, MEy, OPx, OPy)
                elif self.policy_defense == 1:
                    target = (OPx, OPy)
                    if (MEx, MEy) != target:
                        return self.move_to_location(MEx, MEy, *target)
                    else:
                        return self.move_to_location(MEx, MEy, OPx, OPy)
                elif self.policy_defense == 2:
                    target = (OPx + 1, OPy + 1)
                    if (MEx, MEy) != target:
                        return self.move_to_location(MEx, MEy, *target)
                    else:
                        return self.move_to_location(MEx, MEy, OPx, OPy)

    def move_to_location(self, x, y, x_target, y_target):
        if (x, y) == (x_target, y_target):
            raise ValueError(f'invalid move: ({x}, {y}) to target ({x_target}, {y_target})')
        return self.direction(x, y, x_target, y_target)

    def direction(self, x_dep, y_dep, x_dst, y_dst):
        if x_dep == x_dst and y_dep > y_dst:
            return TOP
        elif x_dep < x_dst and y_dep > y_dst:
            return TOP_RIGHT
        elif x_dep < x_dst and y_dep == y_dst:
            return RIGHT
        elif x_dep < x_dst and y_dep < y_dst:
            return BOTTOM_RIGHT
        elif x_dep == x_dst and y_dep < y_dst:
            return BOTTOM
        elif x_dep > x_dst and y_dep < y_dst:
            return BOTTOM_LEFT
        elif x_dep > x_dst and y_dep == y_dst:
            return LEFT
        elif x_dep > x_dst and y_dep > y_dst:
            return TOP_LEFT
        else:
            raise ValueError(
                f'No such direction! (x_dep, y_dep) = ({x_dep}, {y_dep}) (x_dst, y_dst) = ({x_dst}, {y_dst})')

    def move_to_row(self, y, target_row):
        if y < target_row:
            return BOTTOM
        elif y > target_row:
            return TOP
        else:
            return LEFT
