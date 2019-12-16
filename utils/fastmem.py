import torch
import numpy as np

import random
from timebudget import timebudget

class FastMemory:
    def __init__(self, desc, chunks, device):
        self.memory = []
        self.desc = desc
        self.capacity = self.desc.optim_pool_size
        self.chunks = chunks
        self.device = device

    def push(self, episode):
        self._clean()
        assert episode[0].shape[0] == sum(self.chunks), "--> {} {}".format(episode.shape, self.chunks)
        self.memory.extend(episode)

    def sample(self):
        with timebudget("FastMemory-sample"):
            samples = torch.stack(random.sample(self.memory, min(len(self.memory)-1, self.desc.optim_batch_size)))
            for _ in range(self.desc.optim_epochs):
                idx = random.sample(range(len(samples)), min(len(samples)-1, self.desc.batch_size))
                yield [
                        samples[idx, sum(self.chunks[:i+1]):sum(self.chunks[:i+2])
                    ].to(self.device) for i in range(len(self.chunks)-1) ]

    def __len__(self):
        return len(self.memory)

    def _clean(self):
        if len(self.memory) < self.capacity:
            return

        n_clean = len(self.memory) // 4
        del self.memory[:n_clean]
