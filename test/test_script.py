from env import SoccerEnv
from agents.common.training_opponent import StationaryOpponent, RandomSwitchOpponent, RLBasedOpponent

TOP = 0
TOP_RIGHT = 1
RIGHT = 2
BOTTOM_RIGHT = 3
BOTTOM = 4
BOTTOM_LEFT = 5
LEFT = 6
TOP_LEFT = 7

env = SoccerEnv()
agentOP = StationaryOpponent(env_width=env.width, env_height=env.height, env_goal_size=env.goal_size)

state = env.reset()

# loop
env.show()
actionOP = agentOP.get_action(state)
print(actionOP)
done, reward_l, reward_r, state, actions = env.step("type action here!", actionOP)

agentOP.adjust(done, reward_r, i)
