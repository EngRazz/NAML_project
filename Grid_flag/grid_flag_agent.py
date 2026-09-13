"""Flag collection uses the shared tabular Q-learning/SARSA implementation."""
from rl_project.tabular import TabularAgent


def obs_to_key(obs):
    return tuple(int(x) for x in obs['agent']) + tuple(int(x) for x in obs['flags'])


class GridFlagAgent(TabularAgent):
    pass
