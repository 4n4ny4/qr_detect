from typing import Any, Dict, Optional, Tuple
from transformers.cache_utils import DynamicCache
import torch

class DynamicCacheWithQuery(DynamicCache):
    '''
    Cache class used for QRRetriever
    '''
    def __init__(self, query_indices=[]) -> None:
        super().__init__()
        if not hasattr(self, "_seen_tokens"):
            self._seen_tokens = 0
        if not hasattr(self, "key_cache"):
            self.key_cache = []
        if not hasattr(self, "value_cache"):
            self.value_cache = []
        self._query_indices = query_indices # indices for query vectors to save
        self.query_cache = []
        self._max_cache_length = None
        self._cache_lengths = []

    def reserve(self, max_cache_length: int) -> None:
        if self._max_cache_length is None:
            self._max_cache_length = max_cache_length
        else:
            self._max_cache_length = max(self._max_cache_length, max_cache_length)

    def truncate(self, length: int) -> None:
        self._seen_tokens = length
        for layer_idx in range(len(self.key_cache)):
            if layer_idx < len(self._cache_lengths):
                self._cache_lengths[layer_idx] = min(length, self.key_cache[layer_idx].shape[-2])
            elif self._max_cache_length is None:
                self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :, :length, :]
                self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :, :length, :]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx < len(self._cache_lengths):
            return self._cache_lengths[layer_idx]
        if layer_idx < len(self.key_cache):
            return self.key_cache[layer_idx].shape[-2]
        return 0

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        return self.get_seq_length(layer_idx)
    
    def update(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            query_states (`torch.Tensor`):
                The new query states to cache.
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if self._max_cache_length is not None:
            while len(self._cache_lengths) <= layer_idx:
                self._cache_lengths.append(0)

            current_length = self._cache_lengths[layer_idx]
            next_length = current_length + key_states.shape[-2]
            if len(self.key_cache) <= layer_idx:
                cache_shape = (*key_states.shape[:-2], self._max_cache_length, key_states.shape[-1])
                self.key_cache.append(torch.empty(cache_shape, dtype=key_states.dtype, device=key_states.device))
                self.value_cache.append(torch.empty(cache_shape, dtype=value_states.dtype, device=value_states.device))
            elif self.key_cache[layer_idx].shape[-2] < next_length:
                cache_shape = (*key_states.shape[:-2], max(self._max_cache_length, next_length), key_states.shape[-1])
                old_key_cache = self.key_cache[layer_idx]
                old_value_cache = self.value_cache[layer_idx]
                self.key_cache[layer_idx] = torch.empty(cache_shape, dtype=key_states.dtype, device=key_states.device)
                self.value_cache[layer_idx] = torch.empty(cache_shape, dtype=value_states.dtype, device=value_states.device)
                self.key_cache[layer_idx][:, :, :current_length, :].copy_(old_key_cache[:, :, :current_length, :])
                self.value_cache[layer_idx][:, :, :current_length, :].copy_(old_value_cache[:, :, :current_length, :])

            self.key_cache[layer_idx][:, :, current_length:next_length, :].copy_(key_states)
            self.value_cache[layer_idx][:, :, current_length:next_length, :].copy_(value_states)
            self._cache_lengths[layer_idx] = next_length
        elif len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        
        if query_states is not None:
            if len(self.query_cache) <= layer_idx:
                self.query_cache.append(query_states)
            else:
                self.query_cache[layer_idx] = torch.cat([self.query_cache[layer_idx], query_states], dim=-2)
        cache_length = self.get_seq_length(layer_idx)
        return self.key_cache[layer_idx][:, :, :cache_length, :], self.value_cache[layer_idx][:, :, :cache_length, :]
    
    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(None, key_states, value_states, layer_idx)
        return cache
