from training_opponent import TrainingOpponent

class RandomSwitchOpponent(TrainingOpponent):
    def __init__(self, type, env_width, env_height, episode_reset=6):
        super().__init__(type, env_width, env_height)
        self.episode_reset = episode_reset
