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
    def __init__(self, type_attack, type_defense, env_width, env_height, env_goal_size, randomness):
        # attack
        # type 0 | type 1 | type 2 | type 3 | type 4
        #-------------------------------------------
        # xxxxx  | .....  | .....  | .....  | .....
        # x...x  | xxxxx  | .....  | .....  | .....
        # ....x  | ....x  | xxxxx  | ....x  | ....x
        # .....  | .....  | .....  | xxxxx  | x...x
        # .....  | .....  | .....  | .....  | xxxxx
        self.type_attack = type_attack if type_attack else random.randint(0, 4)

        # defense
        # type 0: defense top
        # type 1: defense middle
        # type 2: defense bottom
        self.type_defense = type_defense if type_defense else random.randint(0, 2)

        self.env_width = env_width
        self.env_height = env_height
        self.env_goal_size = env_goal_size

        self.random_attack, self.random_defense = randomness
        if not 0 <= self.random_attack <= 1 or 0 <= self.random_defense <= 1:
            raise ValueError('randomness shold be between 0 and 10')

    def get_action(self, state):
        x_op, y_op, x, y, ball_possession = state

        # random action
        if (ball_possession == 1 and random.random() < self.random_attack) or \
            (ball_possession == 0 and random.random() < self.random_defense):
            print('random action on opponent')
            return random.randint(0, 7)

        # with the ball, attack
        if ball_possession == 1:
            # at the start column and in the middle columns
            if 0 < x <= self.env_width - 1:
                if self.type_attack == 0:
                    return self.move_to_row(y, 0)
                elif self.type_attack == 1:
                    return self.move_to_row(y, int(self.env_height/4))
                elif self.type_attack == 2:
                    return self.move_to_row(y, int(self.env_height/2))
                elif self.type_attack == 3:
                    return self.move_to_row(y, int(self.env_height/4*3))
                elif self.type_attack == 4:
                    return self.move_to_row(y, self.env_height-1)
            # at the end column
            elif x == 0:
                if self.type_attack == 0:
                    return self.move_to_row(y, (self.env_height-self.env_goal_size)/2)
                elif self.type_attack == 1:
                    return self.move_to_row(y, int((2*self.env_height-self.env_goal_size)/4))
                elif self.type_attack == 2:
                    return self.move_to_row(y, int(self.env_height/2))
                elif self.type_attack == 3:
                    return self.move_to_row(y, int((2*self.env_height+self.env_goal_size)/4))
                elif self.type_attack == 4:
                    return self.move_to_row(y, (self.env_height+self.env_goal_size)/2-1)
            else:
                raise ValueError(f'`x` has invalid value `{x}`')
        # without the ball, defense
        else:
            if self.type_defense == 0:
                target = (x_op+1, y_op-1)
                if (x, y) != target:
                    return self.move_to_location(x, y, x_op+1, y_op-1)
                else:
                    return self.move_to_location(x, y, x_op, y_op)
            elif self.type_defense == 1:
                target = (x_op+1, y_op)
                if (x, y) != target:
                    return self.move_to_location(x, y, x_op+1, y_op)
                else:
                    return self.move_to_location(x, y, x_op, y_op)
            elif self.type_defense == 2:
                target = (x_op+1, y_op+1)
                if (x, y) != target:
                    return self.move_to_location(x, y, x_op+1, y_op+1)
                else:
                    return self.move_to_location(x, y, x_op, y_op)

    def move_to_row(self, y, target_row):
        if y < target_row:
            return BOTTOM
        elif y > target_row:
            return TOP
        else:
            return LEFT

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
            raise ValueError(f'No such direction! (x_dep, y_dep) = ({x_dep}, {y_dep}) (x_dst, y_dst) = ({x_dst}, {y_dst})')


class StationaryOpponent(TrainingOpponent):
    def __init__(self, env_width, env_height, env_goal_size, type_attack=None, type_defense=None, randomness=(0.1, 0.1)):
        super().__init__(type_attack, type_defense, env_width, env_height, env_goal_size, difficulty)
        print('StationaryOpponent created')
        print(f'type_attack: {self.type_attack}')
        print(f'type_defense: {self.type_defense}')

    def adjust(self, done, reward, episode_num):
        pass


class RandomSwitchOpponent(TrainingOpponent):
    def __init__(self, env_width, env_height, env_goal_size, type_attack=None, type_defense=None, randomness=(0.1, 0.1), episode_reset=6):
        super().__init__(type_attack, type_defense, env_width, env_height, env_goal_size, difficulty)
        self.episode_reset = episode_reset
        print('RandomSwitchOpponent created')
        print(f'initial type_attack: {self.type_attack}')
        print(f'initial type_defense: {self.type_defense}')

    def adjust(self, done, reward, episode_num):
        if episode_num % self.episode_reset == 0 and done:
            candidate = [type for type in range(5)]
            candidate.remove(self.type_attack)
            self.type_attack = random.choice(candidate)


class RLBasedOpponent(TrainingOpponent):
    def __init__(self, env_width, env_height, env_goal_size, type_attack=None, type_defense=None, randomness=(0.1, 0.1)):
        super().__init__(type_attack, type_defense, env_width, env_height, env_goal_size, difficulty)
        print('RLBasedOpponent created')
        print(f'initial type_attack: {self.type_attack}')
        print(f'initial type_defense: {self.type_defense}')

    def adjust(self, done, reward, episode_num):
        if reward < 0 and done:
            candidate = [type for type in range(5)]
            candidate.remove(self.type_attack)
            self.type_attack = random.choice(candidate)
