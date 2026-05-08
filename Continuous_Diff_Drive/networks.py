import numpy as np
import torch
import torch.nn as nn


class Actor(nn.Module):
    """
    The Actor maps an observation to a deterministic action.

    Architecture: obs → FC → ReLU → FC → ReLU → FC → Tanh → scaled action

    The tanh output is in [-1, 1] and is then linearly scaled to the
    environment's action bounds [action_low, action_high].
    """

    def __init__(self, obs_dim: int, action_dim: int, action_low, action_high, hidden_dim: int = 256):
        """
        Args:
            obs_dim:     Size of the observation vector.
            action_dim:  Size of the action vector.
            action_low:  Lower bound of each action dimension (numpy array).
            action_high: Upper bound of each action dimension (numpy array).
            hidden_dim:  Number of neurons in each hidden layer.
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),   # output in [-1, 1]
        )

        # Register action scaling as buffers so they move with .to(device)
        # register_buffer saves the scale and the bias as constant in the class, will be visible even when change device for computation
        action_scale  = torch.FloatTensor((action_high - action_low) / 2.0)
        action_bias   = torch.FloatTensor((action_high + action_low) / 2.0)
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias",  action_bias)
        #nb actual_action = (output of the network) * action_scale + action_bias --> adapt [-1,1] rage into our range of possible actions

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch_size, obs_dim)
        Returns:
            action: (batch_size, action_dim) — scaled to [action_low, action_high]
        """
        return self.net(obs) * self.action_scale + self.action_bias
        #nb actual_action = (output of the network) * action_scale + action_bias --> adapt [-1,1] rage into our range of possible actions



class Critic(nn.Module):
    """
    The Critic estimates Q(state, action) — how good a (state, action) pair is.

    Architecture:
        obs  → FC → ReLU ─┐
                           cat → FC → ReLU → FC → Q-value (scalar)
        action ────────────┘

    The observation is processed by one layer before being concatenated
    with the action. This is the standard TD3/DDPG critic structure.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Args:
            obs_dim:    Size of the observation vector.
            action_dim: Size of the action vector.
            hidden_dim: Number of neurons in each hidden layer.
        """
        super().__init__()

        # First layer processes the observation only
        self.obs_layer = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
        )

        # Remaining layers process obs features + action together
        self.joint_net = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),   # scalar Q-value
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs:    (batch_size, obs_dim)
            action: (batch_size, action_dim)
        Returns:
            q_value: (batch_size, 1)
        """
        obs_features = self.obs_layer(obs) #compute features from the states
        x = torch.cat([obs_features, action], dim=1) #concatenate result with the action chosed
        return self.joint_net(x) #evaluate the action on the extracted featur from observations


class OUNoise:
    """
    Ornstein-Uhlenbeck process for temporally correlated exploration noise.
    
    dx = θ * (μ - x) * dt + σ * dW
    
    where:
        μ  = mean (usually 0)
        θ  = how fast noise reverts to mean
        σ  = noise magnitude
        dW = Wiener process (gaussian noise)
    """

    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu    = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset noise to mean. Call at the start of each episode."""
        self.state = np.full(self.action_dim, self.mu, dtype=np.float32)

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state.copy()

# ------------------------------------------------------------------
# Weight initialisation helper
# ------------------------------------------------------------------

def init_weights(module: nn.Module, std: float = 0.1):
    """
    Initialise linear layers with small random weights.
    Good initialisation helps DDPG training stability.
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=std)
        nn.init.constant_(module.bias, 0.0)
