from training_opponent import TrainingOpponent

class StationaryOpponent(TrainingOpponent):
    def __init__(self, env_width, env_height, env_goal_size, type=None):
        super().__init__(type, env_width, env_height, env_goal_size)

    def adjust(done, reward, episode_num):
        pass
