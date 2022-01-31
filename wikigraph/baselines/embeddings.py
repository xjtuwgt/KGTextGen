from typing import List, Optional
from torch import nn
import numpy as np

def get_pos_start(timesteps: int, batch_size: int) -> np.ndarray:
  """Find the right slice of positional embeddings for incremental sampling."""
  # pos_start = hk.get_state(
  #     'cache_progress_idx', [batch_size], dtype=jnp.int32, init=jnp.zeros)
  # hk.set_state('cache_progress_idx', pos_start + timesteps)
  pos_start = None
  return pos_start

class SinusoidalPositionEmbedding(nn.Module):
    """Position encoding, using mixture of sinusoidal signals."""
    def __init__(self,
                 dim: int,
                 cache_steps: int = 0,
                 reverse_order: bool = False,
                 clamp_len: Optional[int] = None):
        """Initialize a SinusoidalPositionEmbedding.
      Args:
        dim: Embedding dimension.
        cache_steps: The length of the memory.
        reverse_order: If set to True, position index is reversed.
        clamp_len: position beyond clamp_len will be reset to clamp_len, default
          to not clamping.
      """
        super(SinusoidalPositionEmbedding, self).__init__()
        self._dim = dim
        self._cache_steps = cache_steps
        self._reverse_order = reverse_order
        self._clamp_len = clamp_len
        self._inv_freq = 1.0 / (
                10000 ** (np.arange(0, dim, 2).astype(np.float32) / dim))


    def forward(self, timesteps: int, batch_size: int) -> np.ndarray:
        """Computes the sinusoidal position embedding.
        Args:
          timesteps: The length of the sequence.
          batch_size: The size of the batch.
        Returns:
          Sinusoidal position embedding.
        """
        full_length = timesteps + self._cache_steps

        if self._reverse_order:
          positions = np.arange(full_length - 1, -1, -1)
          positions = np.repeat(positions[None, :], batch_size, axis=0)
        else:
          if self._cache_steps > 0:
            positions = (get_pos_start(timesteps, batch_size)[:, None]
                         + np.arange(timesteps)[None, :])
          else:
            positions = np.arange(0, full_length)
            positions = np.repeat(positions[None, :], batch_size, axis=0)

        if self._clamp_len is not None:
          positions = np.minimum(positions, self._clamp_len)

        scaled_time = positions[:, :, None] * self._inv_freq[None, None, :]
        return np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=2)

