import math
import random

TOP = 0
TOP_RIGHT = 1
RIGHT = 2
BOTTOM_RIGHT = 3
BOTTOM = 4
BOTTOM_LEFT = 5
LEFT = 6
TOP_LEFT = 7


class TrainingOpponent:
    def __init__(self, type, env_width, env_height, env_goal_size):
        self.type = type if type else random.randint(0, 4)
        # type 0 | type 1 | type 2 | type 3 | type 4
        #-------------------------------------------
        # xxxxx  | .....  | .....  | .....  | .....
        # x...x  | xxxxx  | .....  | .....  | .....
        # ....x  | ....x  | xxxxx  | ....x  | ....x
        # .....  | .....  | .....  | xxxxx  | x...x
        # .....  | .....  | .....  | .....  | xxxxx
        self.env_width = env_width
        self.env_height = env_height
        self.env_goal_size = env_goal_size

    def get_action(self, state):
        x_op, y_op, x, y, ball_possession = state

        if ball_possession == 1:
            # with the ball, try to goal

            # at the start column and in the middle columns
            if 0 < x <= self.env_width - 1:
                if self.type == 0:
                    return self.move_to_row(y, 0)
                elif self.type == 1:
                    return self.move_to_row(y, int(self.env_height/4))
                elif self.type == 2:
                    return self.move_to_row(y, int(self.env_height/2))
                elif self.type == 3:
                    return self.move_to_row(int(self.env_height/4*3))
                elif self.type == 4:
                    return self.move_to_row(self.env_height-1)
            # at the end column
            elif x == 0:
                if self.type == 0:
                    return self.move_to_row(y, (self.env_height-self.env_goal_size)/2)
                elif self.type == 1:
                    return self.move_to_row(y, int((2*self.env_height-self.env_goal_size)/4))
                elif self.type == 2:
                    return self.move_to_row(y, int(self.env_height/2))
                elif self.type == 3:
                    return self.move_to_row(y, int((2*self.env_height+self.env_goal_size)/4))
                elif self.type == 4:
                    return self.move_to_row(y, (self.env_height+self.env_goal_size)/2-1)
            else:
                raise ValueError(f'`x` has invalid value `{x}`')
        else:
            # without the ball, chase it
            return self.chase_ball(x_op, y_op, x, y)

    def chase_ball(self, x_op, y_op, x, y):
        if x == x_op and y > y_op:
            return TOP
        elif x < x_op and y > y_op:
            return TOP_RIGHT
        elif x < x_op and y == y_op:
            return RIGHT
        elif x < x_op and y < y_op:
            return BOTTOM_RIGHT
        elif x == x_op and y < y_op:
            return BOTTOM
        elif x > x_op and y < y_op:
            return BOTTOM_LEFT
        elif x > x_op and y == y_op:
            return LEFT
        elif x > x_op and y > y_op:
            return TOP_LEFT
        else:
            raise ValueError(f'location invalid: op: ({x_op}, {y_op}), me: ({x}, {y})')

    def move_to_row(self, y, target_row):
        if y < target_row:
            return BOTTOM
        elif y > target_row:
            return TOP
        else:
            return LEFT


class StationaryOpponent(TrainingOpponent):
    def __init__(self, env_width, env_height, env_goal_size, type=None):
        super().__init__(type, env_width, env_height, env_goal_size)

    def adjust(self, done, reward, episode_num):
        pass


class RandomSwitchOpponent(TrainingOpponent):
    def __init__(self, env_width, env_height, env_goal_size, type=None, episode_reset=6):
        super().__init__(type, env_width, env_height, env_goal_size)
        self.episode_reset = episode_reset

    def adjust(self, done, reward, episode_num):
        if episode_num % episode_reset == 0 and done:
            candidate = [type for type in range(5)].remove(self.type)
            self.type = random.choice(candidate)


class RLBasedOpponent(TrainingOpponent):
    def __init__(self, env_width, env_height, env_goal_size, type=None):
        super().__init__(type, env_width, env_height, env_goal_size)

    def adjust(self, done, reward, episode_num):
        if reward < 0 and done:
            candidate = [type for type in range(5)].remove(self.type)
            self.type = random.choice(candidate)
