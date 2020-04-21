from Module.soccer_env import SoccerEnv

from MADDPG.maddpg import MADDPG
from Module import agent

# set environment
env = SoccerEnv(width=5, height=5, goal_size=3)
env.reset()

# set agents
agent_1 = MADDPG(act_dim=env.act_dim, env_dim=env.env_dim)
agent_2 = agent.opponent()

EPISODES = 1000

for i in range(EPISODES):
    env.reset()

    done = False
    while not done:
        env.show()
