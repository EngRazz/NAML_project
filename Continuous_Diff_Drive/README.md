# Continuous Diff Drive

This folder contains a continuous-control reinforcement learning experiment for a differential-drive robot. The objective is to train a circular robot to navigate a 2D room, avoid walls and rectangular obstacles using LiDAR readings, and reach a goal position. The current learning algorithm is DDPG, which is suitable for continuous actions such as linear and angular velocity.

The project includes the environment, neural network models, replay buffer, DDPG agent, training notebook, evaluation notebook, saved checkpoint, plots, and generated videos.

## Environment

The environment is implemented in `diff_drive_env.py` as a Gymnasium-compatible environment called `DiffDriveEnv`.

The world is a rectangular 2D room. The robot is a circular differential-drive agent with radius `0.3 m`. Obstacles are axis-aligned rectangles represented as:

```python
(x, y, width, height)
```

where `(x, y)` is the bottom-left corner in metres. Walls are also treated as obstacles for collision and LiDAR sensing.

### True State

The full simulator state contains:

- robot position `(x, y)`;
- robot heading `theta`;
- goal position;
- obstacle layout;
- current step count;
- previous distance to the goal;
- previous LiDAR readings used for reward shaping.

The agent does not receive all of this directly. In particular, it does not receive the obstacle map. It must infer nearby obstacles from LiDAR.

### Observation

The observation is a continuous vector of length `n_lidar_rays + 2`. With the default `n_lidar_rays = 16`, the observation has 18 values:

```text
[lidar_0, lidar_1, ..., lidar_15, distance_to_goal, relative_angle_to_goal]
```

The 16 LiDAR readings rotate with the robot heading and measure the distance to the nearest wall or obstacle along evenly spaced rays. Each reading is clipped by `lidar_max_range`, which is `5.0 m` by default.

The last two values are:

- `distance_to_goal`: Euclidean distance from the robot to the goal.
- `relative_angle_to_goal`: angle from the robot heading to the goal direction, wrapped to `[-pi, pi]`.

This observation design is intentionally compact. The robot receives enough information to learn goal-directed navigation and local obstacle avoidance, but it is not given a full map. This forces the policy to use LiDAR rather than memorize obstacle coordinates.

### Actions

The action is continuous and has two values:

```text
[v_linear, v_angular]
```

The bounds are:

- `v_linear` in `[-0.5, 1.0] m/s`;
- `v_angular` in `[-pi, pi] rad/s`.

This action space matches a differential-drive robot at the control level used by the simulator: one command controls forward/backward motion, and the other controls rotation. It is continuous, so value-table methods such as tabular Q-learning are not appropriate.

### Dynamics

At each step, the robot heading is updated by:

```python
theta = theta + v_angular * dt
```

Then the position is updated by moving along the new heading:

```python
x = x + v_linear * cos(theta) * dt
y = y + v_linear * sin(theta) * dt
```

The default time step is `dt = 0.1 s`. If the proposed new position collides with a wall or obstacle, the episode terminates with a collision penalty.

### Reward

The reward is shaped to encourage both reaching the goal and using LiDAR for obstacle-aware navigation.

Terminal rewards:

- `+100` when the goal is reached;
- `-50` on collision.

For non-terminal steps, the reward combines:

- goal progress: positive when the robot gets closer to the goal;
- LiDAR danger reduction: positive when the robot moves from a dangerous clearance to a safer one;
- danger penalty: negative when the robot is too close to obstacles;
- front-sector penalty: negative when obstacles are too close in front of the robot;
- orientation reward: small positive reward for facing the goal;
- time penalty: small negative reward per step.

The key idea is that LiDAR shaping is danger-gated. The robot should not be rewarded simply for moving far away from all obstacles forever. Instead, the LiDAR terms matter most when the robot is close to obstacles or when the direct route to the goal is blocked.

When the LiDAR ray closest to the goal direction is blocked, the environment reduces the goal-progress weight. This helps the robot learn detours: sometimes a good action may temporarily fail to reduce the straight-line distance to the goal because the direct diagonal path is blocked.

### Random Obstacles

When `random_obst=True`, the environment samples obstacle layouts using a curriculum. The default mix is:

- `evaluate_like_detour`: jittered versions of the fixed evaluation layout, including blocked-diagonal cases;
- `wall_with_gap`: long walls split by a passable gap;
- `random_blocks`: smaller random rectangles.

Before accepting a sampled layout, the environment runs a lightweight grid-based feasibility check. Obstacles are inflated by the robot radius, and a BFS search checks that a path from start to goal exists. This avoids training on impossible maps.

## DDPG Agent

The agent is implemented in `diff_drive_agent.py` as `DiffDriveAgent`.

It uses DDPG, an actor-critic algorithm for continuous action spaces. The agent contains:

- an actor network;
- an actor target network;
- a critic network;
- a critic target network;
- an Ornstein-Uhlenbeck noise process for exploration;
- a replay buffer.

### Actor Network

The actor is defined in `networks.py`. It maps an observation to a deterministic action:

```text
observation -> Linear -> ReLU -> Linear -> ReLU -> Linear -> Tanh -> scaled action
```

The final `Tanh` outputs values in `[-1, 1]`. These are then scaled to the environment action bounds. This structure is simple, fast, and appropriate for low-dimensional continuous control. The actor learns the policy: given the current observation, choose the action that should maximize expected return.

### Critic Network

The critic estimates:

```text
Q(observation, action)
```

It first processes the observation, then concatenates the extracted observation features with the action:

```text
observation -> Linear -> ReLU
                         concat with action -> Linear -> ReLU -> Linear -> Q-value
```

The critic learns how good an action is in a given state. The actor is trained using the critic: it changes its actions to maximize the critic's predicted Q-value.

### Target Networks

DDPG uses target networks to stabilize training:

- `actor_target`;
- `critic_target`.

These are slow-moving copies of the online actor and critic. They are used to compute the Bellman target, so the target does not change too abruptly after every gradient update.

The update is a soft update:

```python
target = tau * online + (1 - tau) * target
```

### Replay Buffer

The replay buffer is implemented in `replay_buffer.py`.

It stores transitions:

```text
(state, action, reward, next_state, done)
```

DDPG is off-policy, so it can learn from older experience. The replay buffer is necessary because consecutive environment steps are highly correlated. If the neural networks trained only on the most recent transition, learning would be unstable and inefficient.

The replay buffer solves this by:

- storing many transitions;
- sampling random mini-batches;
- reusing experience many times;
- breaking temporal correlation between updates.

The buffer is circular: when it becomes full, new transitions overwrite the oldest ones.

### Exploration

During training, the actor's deterministic action is perturbed with Ornstein-Uhlenbeck noise. This produces temporally correlated exploration, which is often useful in physical control problems where smooth action changes are more natural than independent random noise at each step.

During evaluation, noise is disabled and the actor is used greedily.

## DDPG Algorithm

DDPG stands for Deep Deterministic Policy Gradient. It combines ideas from deterministic policy gradients, actor-critic learning, deep neural networks, replay buffers, and target networks.

It is designed for continuous action spaces. Instead of evaluating every possible action, the actor directly outputs one action.

### Pseudocode

```text
Initialize actor network µ(s)
Initialize critic network Q(s, a)
Initialize target actor µ_target with actor weights
Initialize target critic Q_target with critic weights
Initialize replay buffer

for each episode:
    reset environment
    reset exploration noise
    observe state s

    while episode is not done:
        if warmup is active:
            choose random action a
        else:
            choose action a = µ(s) + exploration_noise

        execute action a in environment
        observe reward r, next state s_next, and done flag
        store (s, a, r, s_next, done) in replay buffer

        if replay buffer has enough samples:
            sample random batch from replay buffer

            for each transition in batch:
                a_next = µ_target(s_next)
                y = r + gamma * (1 - done) * Q_target(s_next, a_next)

            update critic by minimizing:
                mean squared error between Q(s, a) and y

            update actor using policy gradient:
                maximize Q(s, µ(s))

            softly update target networks:
                target = tau * online + (1 - tau) * target

        s = s_next
```

### Advantages

- Works with continuous action spaces.
- Learns a deterministic policy, so evaluation is simple and fast.
- Replay buffer improves sample reuse.
- Target networks improve stability compared with directly bootstrapping from rapidly changing networks.
- Well suited to low-dimensional robot control tasks like velocity control.

### Limitations

- Sensitive to hyperparameters, reward scaling, and exploration noise.
- Can learn local shortcuts that fail on obstacle layouts not seen during training.
- A deterministic policy may explore poorly in environments with narrow passages.
- It has only one critic, so Q-values can be overestimated.
- It is usually less stable than TD3 or SAC on harder continuous-control problems.


## Folder Structure

```text
Continuous_Diff_Drive/
  diff_drive_env.py      # Gymnasium environment, LiDAR, reward, obstacle sampling, rendering
  diff_drive_agent.py    # DDPG agent, training loop, evaluation loop, plots
  networks.py            # Actor, critic, OU noise, weight initialization
  replay_buffer.py       # Circular replay buffer for off-policy learning
  train_ddpg.ipynb       # Training workflow and random/curriculum evaluation
  eval_ddpg.ipynb        # Fixed-obstacle checkpoint evaluation
  models/                # Saved DDPG checkpoint
  images/                # Training plots
  videos/                # Training and evaluation videos
```

## Notebooks

### `train_ddpg.ipynb`

This notebook is the main training workflow. It:

- configures the environment and DDPG hyperparameters;
- enables curriculum obstacle sampling;
- creates the agent;
- trains the policy;
- saves the checkpoint;
- saves training curves;
- records training and evaluation videos.

Use this notebook when training a new DDPG model.

### `eval_ddpg.ipynb`

This notebook loads the saved checkpoint from:

```text
models/ddpg_checkpoint.pt
```

It evaluates the trained policy on a fixed unseen obstacle layout, including a blocked-diagonal configuration. This is useful for checking whether the agent has learned obstacle-aware detours rather than only moving directly toward the goal.

Use this notebook when testing a trained model without running training again.
