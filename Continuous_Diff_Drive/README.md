# Continuous Diff Drive

This folder contains a continuous-control reinforcement learning experiment for a differential-drive robot. The objective is to train a circular robot to navigate a 2D room, avoid walls and rectangular obstacles using LiDAR readings, and reach a goal position. The folder contains both DDPG and SAC agents so their behavior, training stability, and navigation performance can be compared on the same task.

The project includes the environment, neural network models, replay buffers, DDPG and SAC agents, training notebooks, evaluation notebooks, a DDPG/SAC comparison notebook, saved checkpoints, plots, and generated videos.

## Notebook Index

- `train_ddpg.ipynb`: train DDPG and record DDPG training/evaluation artifacts.
- `eval_ddpg.ipynb`: evaluate the saved DDPG checkpoint on fixed and random obstacle layouts.
- `train_sac.ipynb`: train SAC with the shared environment and record SAC artifacts.
- `eval_sac.ipynb`: evaluate the saved SAC checkpoint on fixed and random obstacle layouts.
- `compare_ddpg_sac.ipynb`: run a direct DDPG-vs-SAC benchmark on the same fixed obstacle layout.

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

- `+500` when the goal is reached;
- `-200` on collision.

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

- `evaluate_like_detour` (`40%`): jittered versions of the fixed evaluation layout, including blocked-diagonal cases;
- `wall_with_gap` (`40%`): long walls split by a passable gap;
- `random_blocks` (`20%`): smaller random rectangles.

Before accepting a sampled layout, the environment runs a lightweight grid-based feasibility check. Obstacles are inflated by `robot_radius + 0.05`, and a BFS search checks that a path from start to goal exists. This avoids training on impossible maps.

## DDPG Agent

The agent is implemented in `diff_drive_agent.py` as `DiffDriveDDPGAgent`.

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


## SAC Agent

The SAC agent is implemented in `diff_drive_agent.py` as `DiffDriveSACAgent`, next to the DDPG `DiffDriveDDPGAgent`. Both agents now use the same `DiffDriveEnv`, observation space, action space, reward, obstacle curriculum, and rendering path. This makes the DDPG/SAC comparison cleaner because algorithm differences are not mixed with environment differences.

SAC stands for Soft Actor-Critic. Like DDPG, it is an off-policy actor-critic method for continuous actions. The main difference is that SAC learns a stochastic policy and explicitly rewards entropy. This means the policy is encouraged to keep exploring useful alternatives instead of becoming deterministic too early.

The SAC agent contains:

- a Gaussian stochastic actor;
- a double critic with two Q-functions;
- a target double critic;
- automatic entropy tuning through a learnable `alpha`;
- a PyTorch replay buffer;
- training and evaluation loops with video recording.

### Gaussian Actor

The actor is defined in `networks.py` as `GaussianActor`. It maps an observation to the parameters of a Gaussian action distribution:

```text
observation -> Linear -> ReLU -> Linear -> ReLU -> mean head
                                                   log_std head
```

The actor outputs:

- `mu`: the mean of the Gaussian policy;
- `log_std`: the log standard deviation, clamped to a safe range.

During training, actions are sampled from this distribution using the reparameterization trick. The sampled action is passed through `tanh` and then rescaled to the environment action bounds. The `tanh` correction is included in the log-probability calculation, which is necessary because squashing changes the probability density.

During evaluation, the actor uses the deterministic mean action. This makes evaluation videos easier to interpret and reduces randomness when comparing SAC with DDPG.

### Double Critic

The critic is defined in `networks.py` as `DoubleCritic`. It contains two separate Q-networks:

```text
Q1(observation, action)
Q2(observation, action)
```

Both critics receive the concatenated observation and action. SAC uses the minimum of the two Q-values when computing targets and policy updates. This is called clipped double-Q learning, and it reduces the positive overestimation bias that can appear when using a single critic.

The SAC implementation also keeps a target double critic. The target critic is a slowly updated copy of the online critic and is used to compute stable Bellman targets.

### Entropy Coefficient

SAC optimizes both reward and entropy. The entropy coefficient `alpha` controls the tradeoff:

- high `alpha`: more exploration and more random actions;
- low `alpha`: more exploitation of the current best policy.

The implementation supports automatic entropy tuning. When enabled, `log_alpha` is learned with its own optimizer so that the policy entropy stays close to a target value. This removes the need to hand-tune a fixed exploration coefficient for every experiment.

### Replay Buffer

The SAC replay buffer is implemented in `replay_buffer.py` as `TorchReplayBuffer`. It stores:

```text
(state, action, reward, next_state, done)
```

SAC is off-policy, so it can reuse old transitions. This is important because neural network updates need random mini-batches rather than highly correlated consecutive transitions. `TorchReplayBuffer` stores data as CPU PyTorch tensors and moves sampled batches to the selected device during sampling, while the original NumPy `ReplayBuffer` remains available for DDPG.

### Shared Environment

SAC uses the same `DiffDriveEnv` as DDPG. There is no separate SAC environment path. This means SAC trains from the same LiDAR-based observation vector and controls the same continuous differential-drive action interface:

```text
observation = [lidar readings, distance_to_goal, relative_goal_angle]
action      = [linear_velocity, angular_velocity]
```

The old experimental SAC reward was preserved as a commented reference block inside `diff_drive_env.py` under `Alternative SAC reward experiment`, but it is not active. This keeps the reward idea available for later design discussion without changing the current shared environment behavior.

## SAC Algorithm

SAC maximizes a soft objective:

```text
expected return + alpha * policy entropy
```

The entropy term rewards policies that remain stochastic. In this navigation task, that is useful because obstacle layouts can require detours, and a deterministic policy may prematurely commit to poor local behaviors. SAC's stochastic policy usually explores more robustly than DDPG, especially in narrow passages or blocked-direct-path scenarios.

### Pseudocode

```text
Initialize Gaussian actor pi(a | s)
Initialize double critic Q1(s, a), Q2(s, a)
Initialize target critics Q1_target, Q2_target
Initialize entropy coefficient alpha, or learn log_alpha automatically
Initialize replay buffer

for each episode:
    reset environment
    observe state s

    while episode is not done:
        if warmup is active:
            choose random action a
        else:
            sample action a from pi(a | s)

        execute action a in environment
        observe reward r, next state s_next, and done flag
        store (s, a, r, s_next, done) in replay buffer

        if replay buffer has enough samples:
            sample random batch from replay buffer

            sample next action a_next from pi(a | s_next)
            compute log probability log_pi(a_next | s_next)

            target_q = r + gamma * (1 - done) *
                       (min(Q1_target(s_next, a_next),
                            Q2_target(s_next, a_next))
                        - alpha * log_pi(a_next | s_next))

            update Q1 and Q2 by minimizing Bellman error

            sample action a_pi from pi(a | s)
            update actor by minimizing:
                alpha * log_pi(a_pi | s) - min(Q1(s, a_pi), Q2(s, a_pi))

            if automatic entropy tuning is enabled:
                update alpha toward the target entropy

            softly update target critics:
                target = tau * critic + (1 - tau) * target

        s = s_next
```

### Advantages

- Works with continuous action spaces.
- Learns a stochastic policy, which usually explores better than DDPG.
- Uses entropy regularization, reducing premature convergence to a narrow policy.
- Uses twin critics, reducing Q-value overestimation.
- Often more stable than DDPG on difficult navigation tasks.

### Limitations

- More complex than DDPG: it has actor, twin critics, target critics, and entropy tuning.
- More hyperparameters and losses must be monitored.
- Training can be slower per update because two critics and stochastic policy log-probabilities are computed.
- The final policy can still fail if the reward or obstacle curriculum does not represent the evaluation scenarios.
- The current SAC code is intended for comparison, but its final performance should be validated with training curves, success rate, and fixed-obstacle evaluation videos.

## Folder Structure

```text
Continuous_Diff_Drive/
  README.md                 # This folder guide
  diff_drive_env.py         # Shared environment, LiDAR, reward, obstacle sampling, rendering
  diff_drive_agent.py       # DDPG and SAC agents, training loops, evaluation loops, plots
  networks.py               # DDPG actor/critic/OU noise and SAC Gaussian actor/double critic
  replay_buffer.py          # DDPG NumPy replay buffer and SAC TorchReplayBuffer

  train_ddpg.ipynb          # DDPG training workflow and random/curriculum evaluation
  eval_ddpg.ipynb           # DDPG fixed-obstacle and random-obstacle checkpoint evaluation
  train_sac.ipynb           # SAC training workflow and deterministic evaluation
  eval_sac.ipynb            # SAC fixed-obstacle and random-obstacle checkpoint evaluation
  compare_ddpg_sac.ipynb    # Direct DDPG vs SAC benchmark on the same fixed map

  models/
    ddpg_checkpoint.pt      # Current DDPG checkpoint
    ddpg_checkpoint_V1.pt   # Earlier DDPG checkpoint snapshot
    sac_checkpoint.pt       # Created after running SAC training

  images/
    ddpg_diff_drive_training_curves.png
    sac_diff_drive_training_curves.png   # Created after running SAC training
    sac_diff_drive_training_curves2.png  # Created after running SAC training

  videos/
    training/               # DDPG training videos
    evaluation/             # DDPG evaluation videos
    training_sac/           # SAC training videos
    evaluation_sac/         # SAC evaluation videos
    comparison/             # Created by the DDPG vs SAC comparison notebook
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

It evaluates the trained policy on a fixed unseen obstacle layout, including a blocked-diagonal configuration, and also includes a random-obstacle evaluation section. This is useful for checking whether the agent has learned obstacle-aware detours rather than only moving directly toward the goal.

Use this notebook when testing a trained model without running training again.

### `train_sac.ipynb`

This notebook is the SAC workflow. It:

- configures the shared environment and SAC hyperparameters;
- creates `DiffDriveSACAgent`;
- trains the Gaussian policy with twin critics;
- saves a SAC checkpoint to `models/sac_checkpoint.pt`;
- records SAC training and evaluation videos;
- saves SAC training plots.

Use this notebook when training the SAC agent for comparison with DDPG. The intended comparison is based on reward curves, success rate, training stability, and performance on fixed obstacle layouts.

### `eval_sac.ipynb`

This notebook loads the saved checkpoint from:

```text
models/sac_checkpoint.pt
```

It evaluates SAC on the same fixed unseen obstacle layout used by `eval_ddpg.ipynb`, then also runs the random-obstacle evaluation section. Videos are saved under:

```text
videos/evaluation_sac/
```

Use this notebook when testing a trained SAC model without running training again.

### `compare_ddpg_sac.ipynb`

This notebook loads both checkpoints:

```text
models/ddpg_checkpoint.pt
models/sac_checkpoint.pt
```

It creates separate DDPG and SAC agents with the same `DiffDriveEnv` configuration, evaluates both greedily on the same fixed obstacle map, records comparison videos, and prints a compact table with reward, steps, final distance, collision status, goal status, and success rate.

Use this notebook when you want an apples-to-apples comparison between the deterministic DDPG policy and the deterministic evaluation path of the SAC policy.
