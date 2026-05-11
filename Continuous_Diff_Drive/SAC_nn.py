import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ============================================================
# Weight Initialization
# ============================================================

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.0)


# ============================================================
# Gaussian Actor (SAC Policy)
# ============================================================

class GaussianActor(nn.Module):
    """
    SAC stochastic policy.

    Maps:
        observation -> Gaussian distribution parameters

    Outputs:
        mu       : mean action
        log_std  : log standard deviation

    Uses:
        - reparameterization trick
        - tanh squashing
        - action rescaling
    """

    LOG_STD_MAX = 2
    LOG_STD_MIN = -20

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_low,
        action_high,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # ========================================================
        # Shared backbone
        # ========================================================

        self.base_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # ========================================================
        # Mean and std heads
        # ========================================================

        self.mu_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

        # ========================================================
        # Action scaling
        # ========================================================

        action_scale = torch.tensor(
            (action_high - action_low) / 2.0,
            dtype=torch.float32
        )

        action_bias = torch.tensor(
            (action_high + action_low) / 2.0,
            dtype=torch.float32
        )

        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias", action_bias)

        self.apply(init_weights)

    # ============================================================
    # Forward
    # ============================================================

    def forward(self, obs: torch.Tensor):

        x = self.base_net(obs)

        mu = self.mu_layer(x)

        log_std = self.log_std_layer(x)

        log_std = torch.clamp(
            log_std,
            min=self.LOG_STD_MIN,
            max=self.LOG_STD_MAX
        )

        return mu, log_std

    # ============================================================
    # Sample action
    # ============================================================

    def sample(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ):
        """
        Returns:
            action
            log_prob
        """

        mu, log_std = self.forward(obs)

        std = log_std.exp()

        normal = Normal(mu, std)

        # ========================================================
        # Reparameterization trick
        # ========================================================

        if deterministic:
            x_t = mu
        else:
            x_t = normal.rsample()

        # ========================================================
        # Tanh squashing
        # ========================================================

        y_t = torch.tanh(x_t)

        # ========================================================
        # Rescale to env action range
        # ========================================================

        action = (
            y_t * self.action_scale +
            self.action_bias
        )

        # ========================================================
        # Log probability correction
        # ========================================================

        if deterministic:

            log_prob = None

        else:

            log_prob = normal.log_prob(x_t)

            # SAC tanh correction
            log_prob -= torch.log(
                self.action_scale * (1 - y_t.pow(2)) + 1e-6
            )

            log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob

    # ============================================================
    # Greedy action
    # ============================================================

    def act(self, obs: torch.Tensor):

        action, _ = self.sample(
            obs,
            deterministic=True
        )

        return action


# ============================================================
# Double Critic
# ============================================================

class DoubleCritic(nn.Module):
    """
    Twin-Q critic used in SAC.

    Contains:
        Q1(s,a)
        Q2(s,a)

    Used to reduce positive bias in value estimates.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        input_dim = obs_dim + action_dim

        # ========================================================
        # Q1
        # ========================================================

        self.q1_net = nn.Sequential(

            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, 1)
        )

        # ========================================================
        # Q2
        # ========================================================

        self.q2_net = nn.Sequential(

            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, 1)
        )

        self.apply(init_weights)

    # ============================================================
    # Forward
    # ============================================================

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ):

        xu = torch.cat([obs, action], dim=1)

        q1 = self.q1_net(xu)

        q2 = self.q2_net(xu)

        return q1, q2

    # ============================================================
    # Optional helper
    # ============================================================

    def q1(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ):

        xu = torch.cat([obs, action], dim=1)

        return self.q1_net(xu)