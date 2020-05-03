import random
import numpy as np

TOP = 0
TOP_RIGHT = 1
RIGHT = 2
BOTTOM_RIGHT = 3
BOTTOM = 4
BOTTOM_LEFT = 5
LEFT = 6
TOP_LEFT = 7

class SoccerEnv():
    def __init__(self, width=5, height=5, goal_size=3, prob_random=0.05):
        # check if the dimension is valid
        if width < 2:
            raise ValueError('`width` must be greater than 2')
        elif not 0 < goal_size <= height:
            raise ValueError('`goal_size` must be greater than 0 and smaller or equal to `height`')
        elif (height - goal_size) % 2 != 0:
            raise ValueError('`height` and `goal_size` must both be odd or even')

        # set parameter of the environment
        self.env_dim = 5 # (agent_left_x, agent_left_y, agent_right_x, agent_right_y, ball_possession)
        self.act_dim = 8 # split 360 degrees by 8 directions by 45-degree difference

        # set dimension of the field
        self.width = width
        self.height = height
        self.goal_size = goal_size

        # probability distribution of actions
        # 1 - prob_random: correct direction
        # 4 : 3 : 2 : 1 = prob of 45, 90, 125, 180 degrees away from correct direction
        # 19 = 4 + 3 + 2 + 1 + 2 + 3 + 4
        # prob_distribution = [correct, 45, 90, 135, 180, -135, -90, -45]
        self.prob_distribution = [1-prob_random] + \
                                 [prob_random/19*i for i in [4, 3, 2, 1, 2, 3, 4]]

        # initialize agents and ball possession
        self.agent_left = Agent()
        self.agent_right = Agent()
        self.ball_possession = None

    def reset(self):
        self.agent_left.set_xy(0, int(self.height/2))
        self.agent_right.set_xy(self.width-1, int(self.height/2))
        # 0 for left possession, 1 for right possession
        self.ball_possession = random.randint(0, 1)

        # state = (agent_left_x, agent_left_y, agent_right_x, agent_right_y, ball_possession)
        state = self.agent_left.get_xy() + self.agent_right.get_xy() + (self.ball_possession,)
        actions = (None, None)

        return state

    def step(self, agent_left_action, agent_right_action):
        # add randomness into the environment
        al_actual_action = self.get_actual_action(agent_left_action)
        ar_actual_action = self.get_actual_action(agent_right_action)

        if al_actual_action != agent_left_action:
            print('env randomness on agent_left')
        if ar_actual_action != agent_right_action:
            print('env randomness on agent_right')

        # check if game is over and the rewards
        done, reward_l, reward_r = self.game_over(al_actual_action, ar_actual_action)

        if not done:
            # underscore (_) after variable means next state
            al_loc_ = self.agent_left.move(al_actual_action)
            ar_loc_ = self.agent_right.move(ar_actual_action)

            # check if next state locations are valid
            # if not, next state location = original location
            if not self.location_valid(al_loc_):
                al_loc_ = self.agent_left.get_xy()
                reward_l -= 1
            if not self.location_valid(ar_loc_):
                ar_loc_ = self.agent_right.get_xy()
                reward_r -= 1

            if self.change_possesion(al_loc_, ar_loc_):
                # switch ball possession
                self.ball_possession = int(not self.ball_possession)
                # give reward to the agent who steal the ball
                if self.ball_possession == 0:
                    # left agent steal the ball
                    reward_l += 2
                    reward_r -= 2
                elif self.ball_possession == 1:
                    # right agent steal the ball
                    reward_l -= 2
                    reward_r += 2
                # if ball possession switched, next state locations = original locations
                al_loc_ = self.agent_left.get_xy()
                ar_loc_ = self.agent_right.get_xy()

            self.agent_left.set_xy(*al_loc_)
            self.agent_right.set_xy(*ar_loc_)

        # state = (agent_left_x, agent_left_y, agent_right_x, agent_right_y, ball_possession)
        state = self.agent_left.get_xy() + self.agent_right.get_xy() + (self.ball_possession,)
        actions = (al_actual_action, ar_actual_action)

        return done, reward_l, reward_r, state, actions

    def location_valid(self, location):
        x, y = location
        if 0 <= x < self.width and 0 <= y < self.height:
            return True
        else:
            return False

    def game_over(self, agent_left_action, agent_right_action):
        # underscore (_) after variable means next state
        al_x_, al_y_ = self.agent_left.move(agent_left_action)
        if self.ball_possession == 0 and \
            al_x_ == self.width and \
            (self.height - self.goal_size) / 2 <= al_y_ <= (self.height + self.goal_size) / 2 - 1:
            # left agent wins
            # return if_game_over, left_reward, right_reward
            return True, 10, -10

        ar_x_, ar_y_ = self.agent_right.move(agent_right_action)
        if self.ball_possession == 1 and \
            ar_x_ == -1 and \
            (self.height - self.goal_size) / 2 <= ar_y_ <= (self.height + self.goal_size) / 2 - 1:
            # right agent wins
            # return if_game_over, left_reward, right_reward
            return True, -10, 10

        # game not end yet
        return False, 0, 0

    def get_actual_action(self, action):
        prob_distribution = self.prob_distribution[-action:] + self.prob_distribution[:-action]
        return np.random.choice([i for i in range(8)], p=prob_distribution)

    def change_possesion(self, al_next_loc, ar_next_loc):
        al_current_loc = self.agent_left.get_xy()
        ar_current_loc = self.agent_right.get_xy()

        if al_current_loc == ar_next_loc and ar_current_loc == al_next_loc:
            return True
        elif al_next_loc == ar_next_loc:
            return True
        else:
            return False

    def show(self):
        if self.ball_possession == 0:
            left = '▲'
            right = '○'
        else:
            left = '△'
            right = '●'

        for y in range(self.height):
            for x in range(-1, self.width+1):
                # draw the goals
                if (x == -1 or x == self.width) and \
                    ((self.height - self.goal_size) / 2 <= y <= (self.height + self.goal_size) / 2 - 1):
                    print('+', end='')
                    continue
                elif x == -1 or x == self.width:
                    print(' ', end='')
                    continue

                if (x, y) == self.agent_left.get_xy():
                    print(left, end='')
                elif (x, y) == self.agent_right.get_xy():
                    print(right, end='')
                else:
                    print('.', end='')
            print()


class Agent():
    def __init__(self, x=None, y=None):
        self.set_xy(x, y)

    def move(self, action):
        moves = {
            TOP         : lambda: (self.x,   self.y-1),
            TOP_RIGHT   : lambda: (self.x+1, self.y-1),
            RIGHT       : lambda: (self.x+1, self.y),
            BOTTOM_RIGHT: lambda: (self.x+1, self.y+1),
            BOTTOM      : lambda: (self.x,   self.y+1),
            BOTTOM_LEFT : lambda: (self.x-1, self.y+1),
            LEFT        : lambda: (self.x-1, self.y),
            TOP_LEFT    : lambda: (self.x-1, self.y-1)
        }

        return moves.get(action, lambda: (self.x, self.y))()

    def set_xy(self, x, y):
        self.x = x
        self.y = y

    def get_xy(self):
        return (self.x, self.y)
