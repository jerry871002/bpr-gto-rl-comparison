import pickle
from soccer_stat import SoccerStat

lst = [1] + [i for i in range(500, 10001, 500)]
dir = 'stats/test/'
agent_type = 'maddpg'

for item in lst:
    filename = f'{agent_type}_{item}.pkl'
    stat = pickle.load(open(dir + filename, 'rb'))
    print(stat.get_win_rate()[0])

agent_type = 'maddpg_d'
print()

for item in lst:
    filename = f'{agent_type}_{item}.pkl'
    stat = pickle.load(open(dir + filename, 'rb'))
    print(stat.get_win_rate()[0])
