import math

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
        self.type = type
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

            # in the middle columns
            if 0 < x < self.env_width - 1:
                return LEFT
            # at the start column
            elif x == self.env_width - 1:
                if self.type == 0:
                    return TOP if y != 0 else LEFT
                elif self.type == 1:
                    return TOP if y != int(self.env_height/4) else LEFT
                elif self.type == 2:
                    return LEFT
                elif self.type == 3:
                    return BOTTOM if y != int(self.env_height/4*3) else LEFT
                elif self.type == 4:
                    return BOTTOM if y != self.env_height - 1 else LEFT
            # at the end column
            elif x == 0:
                if self.type == 0:
                    return BOTTOM if y != (self.env_height-self.env_goal_size)/2 else LEFT
                elif self.type == 1:
                    return BOTTOM if y != int((2*self.env_height-self.env_goal_size)/4) else LEFT
                elif self.type == 2:
                    return LEFT
                elif self.type == 3:
                    return TOP if y != int((2*self.env_height+self.env_goal_size)/4) else LEFT
                elif self.type == 4:
                    return TOP if y != (self.env_height + self.env_goal_size) / 2 else LEFT
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
