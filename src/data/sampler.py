from __future__ import annotations

import random

from torch.utils.data import Sampler


class TokenBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        lengths: list[int],
        max_tokens: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int | None = None,
    ):
        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        self.lengths = lengths
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self._iteration = 0

    def _ordered_indices(self, iteration: int) -> list[int]:
        indices = list(range(len(self.lengths)))
        if not self.shuffle:
            return indices
        seed = None if self.seed is None else self.seed + iteration
        rng = random.Random(seed)
        rng.shuffle(indices)
        return indices

    def _batch_count_for_indices(self, indices: list[int], include_dropped: bool = False) -> int:
        batch_count = 0
        batch = []
        max_len = 0
        for idx in indices:
            length = int(self.lengths[idx])
            next_max_len = max(max_len, length)
            next_cost = next_max_len * (len(batch) + 1)
            if batch and next_cost > self.max_tokens:
                batch_count += 1
                batch = []
                max_len = 0
            batch.append(idx)
            max_len = max(max_len, length)
        if batch and (include_dropped or not self.drop_last):
            batch_count += 1
        return batch_count

    def __iter__(self):
        indices = self._ordered_indices(self._iteration)
        self._iteration += 1

        batch = []
        max_len = 0
        for idx in indices:
            length = int(self.lengths[idx])
            next_max_len = max(max_len, length)
            next_cost = next_max_len * (len(batch) + 1)
            if batch and next_cost > self.max_tokens:
                yield batch
                batch = []
                max_len = 0
            batch.append(idx)
            max_len = max(max_len, length)

        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        if self.shuffle and self.seed is None:
            return len(self.lengths)
        return self._batch_count_for_indices(self._ordered_indices(self._iteration))
