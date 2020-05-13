# stationary, randomswitch and RLbased left agents

import math
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


class TrainingAgent:
    def __init__(self, type_attack, type_defense, env_width, env_height, env_goal_size, randomness):
        # defense
        # type 0 | type 1 | type 2 | type 3 | type 4
        #-------------------------------------------
        # xxxxx  | .....  | .....  | .....  | .....
        # x...x  | xxxxx  | .....  | .....  | .....
        # ....x  | ....x  | xxxxx  | ....x  | ....x
        # .....  | .....  | .....  | xxxxx  | x...x
        # .....  | .....  | .....  | .....  | xxxxx
        self.type_defense = type_defense if type_defense else random.randint(0, 4)

        # attack
        # type 0: defense top
        # type 1: defense middle
        # type 2: defense bottom
        self.type_attack = type_attack if type_attack else random.randint(0, 2)

        self.env_width = env_width
        self.env_height = env_height
        self.env_goal_size = env_goal_size
        self.back_to_origin = True

        self.random_attack, self.random_defense = randomness
        if not (0 <= self.random_attack <= 1 and 0 <= self.random_defense <= 1):
            raise ValueError('randomness should be between 0 and 1')

    def get_action(self, state):
        MEx, MEy, OPx, OPy, possession = state
        # left has ball, attacking
        if possession == 0:
            # in front of goal
            if MEx == self.env_width-1 and MEy == self.env_height-1:
                return TOP_RIGHT
            if MEx == self.env_width-1 and MEy == 0:
                return BOTTOM_RIGHT
            # close to opponent, show policy
            if (abs(MEx - OPx) + abs(MEy - OPy) <= 2 and MEx < OPx):
                if self.type_attack == 0: # attack from top
                    return self.to_target(MEx, MEy, OPx, OPy-1)
                if self.type_attack == 1: # attack from middle
                    return self.to_target(MEx, MEy, OPx, OPy)
                if self.type_attack == 2: # attack from bottom
                    return self.to_target(MEx, MEy, OPx, OPy+1)
            # or (MEx == OPx and MEy == OPy-1) or (MEx == OPx and MEy == OPy+1)
            
            return RIGHT
        # defending
        else:
            # print('back to origin: ', self.back_to_origin)
            #go back to origin first
            if MEx == 0 and MEy == int(self.env_height/2):
                self.back_to_origin = True
            if not self.back_to_origin:
                return self.to_target(MEx, MEy, 0, int(self.env_height/2))
            # try to block OP
            if self.type_defense == 0:
                return self.move_to_row(state[1], 0)
            elif self.type_defense == 1:
                return self.move_to_row(state[1], int(self.env_height/4))
            elif self.type_defense == 2:
                return self.move_to_row(state[1], int(self.env_height/2))
            elif self.type_defense == 3:
                return self.move_to_row(state[1], int(self.env_height/4*3))
            elif self.type_defense == 4:
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


class StationaryAgent(TrainingAgent):
    def __init__(self, env_width, env_height, env_goal_size, type_attack=None, type_defense=None, randomness=(0.1, 0.1)):
        super().__init__(type_attack, type_defense, env_width, env_height, env_goal_size, randomness)
        print('StationaryAgent created')
        print(f'type_attack: {self.type_attack}')
        print(f'type_defense: {self.type_defense}')

    def adjust(self, done, reward, episode_num):
        pass


class RandomSwitchAgent(TrainingAgent):
    def __init__(self, env_width, env_height, env_goal_size, type_attack=None, type_defense=None, randomness=(0.1, 0.1), episode_reset=6):
        super().__init__(type_attack, type_defense, env_width, env_height, env_goal_size, randomness)
        self.episode_reset = episode_reset
        print('RandomSwitchAgent created')
        print(f'initial type_attack: {self.type_attack}')
        print(f'initial type_defense: {self.type_defense}')

    def adjust(self, done, reward, episode_num):
        print(f'episode_num: {episode_num}')
        if (episode_num + 1) % self.episode_reset == 0 and done:
            candidate = [type for type in range(5)]
            candidate.remove(self.type_defense)
            self.type_defense = random.choice(candidate)
            print(f'Agent type_defense switch to {self.type_defense}')
            
            candidate = [type for type in range(3)]
            candidate.remove(self.type_attack)
            self.type_attack = random.choice(candidate)
            print(f'Agent type_attack switch to {self.type_attack}')


class RLBasedAgent(TrainingAgent):
    def __init__(self, env_width, env_height, env_goal_size, type_attack=None, type_defense=None, randomness=(0.1, 0.1)):
        super().__init__(type_attack, type_defense, env_width, env_height, env_goal_size, randomness)
        print('RLBasedAgent created')
        print(f'initial type_attack: {self.type_attack}')
        print(f'initial type_defense: {self.type_defense}')

    def adjust(self, done, reward, episode_num):
        if reward < 0 and done:
            candidate = [type for type in range(5)]
            candidate.remove(self.type_attack)
            self.type_attack = random.choice(candidate)
            print(f'Agent type_attack switch to {self.type_attack}')
            
            candidate = [type for type in range(3)]
            candidate.remove(self.type_defense)
            self.type_defense = random.choice(candidate)
            print(f'Agent type_defense switch to {self.type_defense}')