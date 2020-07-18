from env import SoccerEnv

from agents.BPR_3_attack_3_defense.bpr_3_3 import BPR_3_3
from agents.BPR_3_attack_3_defense.bpr_op_3_3 import BPR_OP_3_3

# set environment
env = SoccerEnv(width=5, height=5, goal_size=3)

# set agents
agentME = BPR_3_3(env_width=env.width, env_height=env.height, env_goal_size=env.goal_size)
agentOP = BPR_OP_3_3(env_width=env.width, env_height=env.height, env_goal_size=env.goal_size)

# parameters
EPISODES = 1000
win = 0
for i in range(EPISODES):
    state = env.reset()
    ball_possession_change = True
    done = False
    while not done:
        leftx, lefty, rightx, righty, possession = state
        env.show()
        print()

        if ball_possession_change:
            agentME.change_policy(leftx, lefty, possession)
            agentOP.change_policy(rightx, righty, possession)

        # agent 1 decides its action
        actionME = agentME.choose_action(state)

        # agent 2 decides its action
        actionOP = agentOP.choose_action(state)

        # perform actions on the environment
        done, reward_l, reward_r, state_, actions = env.step(actionME, actionOP)

        if done and reward_l == 10:
            win += 1

        # actions[0]: action of left agent, actions[1]: action of right agent
        # training process of agent 1
        agentME.update_belief(state, actions[1])

        # training process of agent 2
        agentOP.update_belief(state, actions[0])

        ball_possession_change = not (state[4] == state_[4])
        state = state_

print(f'win rate = {win / EPISODES}')
