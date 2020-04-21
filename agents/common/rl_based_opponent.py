import random

from training_opponent import TrainingOpponent

class RLBasedOpponent(TrainingOpponent):
    def __init__(self, env_width, env_height, env_goal_size, type=None):
        super().__init__(type, env_width, env_height, env_goal_size)

    def adjust(done, reward, episode_num):
        if reward < 0 and done:
            candidate = [type for type in range(5)].remove(self.type)
            self.type = random.choice(candidate)
