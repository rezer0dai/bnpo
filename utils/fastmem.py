import torch
import numpy as np

import random
from timebudget import timebudget

import sys
sys.path.append("PrioritizedExperienceReplay")
from PrioritizedExperienceReplay.proportional import Experience

class FastMemory:
    def __init__(self, desc, chunks, device):
        self.memory = []
        self.desc = desc
        self.capacity = self.desc.optim_pool_size
        self.chunks = chunks
        self.device = device

    def push(self, episode, _):
        self._clean()
        assert episode[0].shape[0] == sum(self.chunks), "--> {} {}".format(episode.shape, self.chunks)
        self.memory.extend(episode)

    def sample(self):
        with timebudget("FastMemory-sample"):
            samples = torch.stack(random.sample(self.memory, min(len(self.memory)-1, self.desc.optim_batch_size)))
            for _ in range(self.desc.optim_epochs):
                idx = random.sample(range(len(samples)), min(len(samples)-1, self.desc.batch_size))
                yield (torch.ones(len(idx)), [
                        samples[idx, sum(self.chunks[:i+1]):sum(self.chunks[:i+2])
                    ].to(self.device) for i in range(len(self.chunks)-1) ])

    def __len__(self):
        return len(self.memory)

    def _clean(self):
        if len(self.memory) < self.capacity:
            return

        n_clean = len(self.memory) // 4
        del self.memory[:n_clean]

class IRFastMemory:
    def __init__(self, desc, chunks, device):
        self.memory = Experience(desc.optim_pool_size, desc.batch_size, .666)
        self.desc = desc
        self.chunks = chunks
        self.device = device

        self.beta = desc.prio_schedule

    def push(self, episode, ir):
        ir = torch.where(ir != 0, ir, torch.zeros_like(ir).fill_(1e-8))
        assert episode[0].shape[0] == sum(self.chunks), "--> {} {}".format(episode.shape, self.chunks)
        for e, r in zip(episode, ir):
            self.memory.add(e, r)

    def sample(self):
        with timebudget("IR_FastMemory-sample"):
            samples, w_is, _ = self.memory.select(self.beta())
            if samples is not None:
                samples = torch.stack(samples)
                w_is = torch.ones(w_is.shape)
                for _ in range(self.desc.optim_epochs):
                    idx = random.sample(range(len(samples)), min(len(samples)-1, self.desc.batch_size))
                    yield (w_is[idx], [
                            samples[idx, sum(self.chunks[:i+1]):sum(self.chunks[:i+2])
                        ].to(self.device) for i in range(len(self.chunks)-1) ])

    def __len__(self):
        return len(self.memory)

class PERFastMemory:
    def __init__(self, desc, chunks, device):
        self.memory = Experience(desc.optim_pool_size, desc.batch_size, .666)
        self.desc = desc
        self.chunks = chunks
        self.device = device

        self.beta = desc.prio_schedule

    def push(self, episode, ir):
        ir = torch.where(ir != 0, ir, torch.zeros_like(ir).fill_(1e-8))
        assert episode[0].shape[0] == sum(self.chunks), "--> {} {}".format(episode.shape, self.chunks)
        for e, r in zip(episode, ir):
            self.memory.add(e, r)

    def sample(self):
        with timebudget("PER_FastMemory-sample"):
            samples, w_is, _ = self.memory.select(self.beta())
            if samples is not None:
                samples = torch.stack(samples)
                w_is = torch.from_numpy(w_is)
                for _ in range(self.desc.optim_epochs):
                    idx = random.sample(range(len(samples)), min(len(samples)-1, self.desc.batch_size))
                    yield (w_is[idx], [
                            samples[idx, sum(self.chunks[:i+1]):sum(self.chunks[:i+2])
                        ].to(self.device) for i in range(len(self.chunks)-1) ])

    def __len__(self):
        return len(self.memory)
