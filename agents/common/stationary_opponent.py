from training_opponent import TrainingOpponent

class StationaryOpponent(TrainingOpponent):
    def __init__(self, type, env_width, env_height):
        super().__init__(type, env_width, env_height)
