import numpy as np


class ReplayBuffer:
    """
    Fixed-size circular replay buffer for DDPG.

    Stores transitions (state, action, reward, next_state, done) and
    returns random batches for training the actor and critic networks.

    Uses pre-allocated numpy arrays for efficiency — no Python lists
    or dynamic memory allocation during training.
    """

    def __init__(self, obs_dim: int, action_dim: int, max_size: int = 100_000):
        """
        Args:
            obs_dim:    Dimensionality of the observation vector (18 for DiffDrive).
            action_dim: Dimensionality of the action vector (2 for DiffDrive).
            max_size:   Maximum number of transitions to store.
                        When full, oldest transitions are overwritten.
        """
        self.max_size   = max_size
        self.ptr        = 0      # points to the next slot to write
        self.size       = 0      # current number of stored transitions

        # Pre-allocate storage
        self.states      = np.zeros((max_size, obs_dim),    dtype=np.float32)
        self.actions     = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards     = np.zeros((max_size, 1),          dtype=np.float32)
        self.next_states = np.zeros((max_size, obs_dim),    dtype=np.float32)
        self.dones       = np.zeros((max_size, 1),          dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, state, action, reward: float, next_state, done: bool):
        """Store a single transition."""
        self.states[self.ptr]      = state
        self.actions[self.ptr]     = action
        self.rewards[self.ptr]     = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr]       = float(done)

        # Circular write — overwrite oldest entry when full
        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> dict:
        """
        Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            A dict with keys: states, actions, rewards, next_states, dones.
            All values are numpy float32 arrays of shape (batch_size, dim).
        """
        assert self.size >= batch_size, (
            f"Buffer has only {self.size} transitions, cannot sample {batch_size}."
        )
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "states"     : self.states[idx],
            "actions"    : self.actions[idx],
            "rewards"    : self.rewards[idx],
            "next_states": self.next_states[idx],
            "dones"      : self.dones[idx],
        }

    def ready(self, batch_size: int) -> bool:
        """Returns True when the buffer has enough transitions to sample."""
        return self.size >= batch_size

    def __len__(self) -> int:
        return self.size
