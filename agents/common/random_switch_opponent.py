import random

from training_opponent import TrainingOpponent

class RandomSwitchOpponent(TrainingOpponent):
    def __init__(self, env_width, env_height, env_goal_size, type=None, episode_reset=6):
        super().__init__(type, env_width, env_height, env_goal_size)
        self.episode_reset = episode_reset

    def adjust(done, reward, episode_num):
        if episode_num % episode_reset == 0 and done:
            candidate = [type for type in range(5)].remove(self.type)
            self.type = random.choice(candidate)
