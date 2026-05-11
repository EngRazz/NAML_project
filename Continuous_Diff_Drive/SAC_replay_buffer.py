import torch

class ReplayBuffer:
    """
    Versione ottimizzata con PyTorch Tensor per eliminare i colli di bottiglia su CPU.
    """

    def __init__(self, obs_dim: int, action_dim: int, max_size: int = 100_000, device: str = "cpu"):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = device

        # Pre-allochiamo tutto come PyTorch Tensor
        self.states      = torch.zeros((max_size, obs_dim),    dtype=torch.float32)
        self.actions     = torch.zeros((max_size, action_dim), dtype=torch.float32)
        self.rewards     = torch.zeros((max_size, 1),          dtype=torch.float32)
        self.next_states = torch.zeros((max_size, obs_dim),    dtype=torch.float32)
        self.dones       = torch.zeros((max_size, 1),          dtype=torch.float32)

    def add(self, state, action, reward, next_state, done):
        # Convertiamo l'input in tensor solo una volta all'ingresso
        # as_tensor non copia se l'input è già compatibile
        self.states[self.ptr]      = torch.as_tensor(state, dtype=torch.float32)
        self.actions[self.ptr]     = torch.as_tensor(action, dtype=torch.float32)
        self.rewards[self.ptr]     = float(reward)
        self.next_states[self.ptr] = torch.as_tensor(next_state, dtype=torch.float32)
        self.dones[self.ptr]       = float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> dict:
        assert self.size >= batch_size, f"Buffer ha solo {self.size} elementi."
        
        # Generiamo indici casuali
        idx = torch.randint(0, self.size, (batch_size,))

        # Restituiamo i dati (niente conversioni qui, sono già Tensor!)
        return {
            "states":      self.states[idx].to(self.device),
            "actions":     self.actions[idx].to(self.device),
            "rewards":     self.rewards[idx].to(self.device),
            "next_states": self.next_states[idx].to(self.device),
            "dones":       self.dones[idx].to(self.device)
        }

    def ready(self, batch_size: int) -> bool:
        return self.size >= batch_size

    def __len__(self) -> int:
        return self.size
