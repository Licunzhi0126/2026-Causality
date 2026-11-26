"""Immutable experiment controls; network seeds never become training seeds."""
from dataclasses import dataclass
from ..formal_protocol import FORMAL_SEED


@dataclass(frozen=True)
class RandomNetworkConfig:
    n_random: int = 100
    base_seed: int = FORMAL_SEED
    train_seed: int = FORMAL_SEED
    k: int = 40
    device: str = 'cpu'
    max_retries: int = 0

    def __post_init__(self):
        if self.n_random < 1 or self.k < 2 or min(self.base_seed, self.train_seed, self.max_retries) < 0:
            raise ValueError('Invalid count, k, seed or retry count')
        if self.device not in {'cpu', 'cuda', 'cuda', 'auto'}:
            raise ValueError('Unsupported device')
