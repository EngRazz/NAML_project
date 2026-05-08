# NAML Project Decisions

## Discreate (Grid) Environment

### Description
**States**: position of the agent, position of the flags to collect
**Agent**: a point that moves on the grid
**Reward**: increases is the agent gets a flag, decreases for every time step

The time to get the flags is limited.

### Algorithms
- **Q-Learnign**
- **SARSA**

### Plots
- Reward(epochs), with confidence interval

### Recordings
- Videos of the some training episodes
- Videos of some evaluation episodes

## Continuos Environment

### Description
**States**: position of the agent, LiDAR observations
**Agent**: a diff-drive circular robot with a LiDAR for sensing
**Reward**: decrease if the agent bumps an obstacle (walls are considered obstacles), increase if it gets the goal position

### Algorithms
- **DDPG**
- **SAC**

### Plots
- Reward(epochs), with confidence interval
- Final Critic Q-Table (maybe???)

### Recordings
- Videos of the some training episodes
- Videos of some evaluation episodes
