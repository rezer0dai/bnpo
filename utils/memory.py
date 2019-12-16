import torch
import numpy as np
import random
import itertools

# we want to store tensors as CPU only i think ..
class Memory:
    def __init__(self, capacity, recalc_delay, chunks, ep_draw, device):
        self.memory = []

        self.ep_draw = ep_draw
        self.chunks = [0] + chunks
        self.back_pointers = []
        self.capacity = capacity
        self.device = device
        self.allowed_mask = []

        self.recalc_delay = recalc_delay

    def push(self, experience, allowed_mask):
        self._clip()
        self.allowed_mask.extend(allowed_mask)
        self._sync(experience)

        self.memory = torch.cat([self.memory, experience]) if len(self.memory) else experience

        assert len(self.memory) == len(self.back_pointers)
        assert len(self.memory) == len(self.allowed_mask)

    def sample(self, update, batch_size, back_view=0):
        idx, eps = self._select(batch_size, back_view)

        if update is not None: # now we getting hairy computationaly
            samples = self._sample_with_update(idx, eps, update)
        else:
            samples = self.memory[list(itertools.chain.from_iterable(idx))]

        return self._decompose(samples[:batch_size].to(self.device))

    def __len__(self):
        if not len(self.back_pointers):
            return 0
        return self.back_pointers[-1][1]

    def _sample_with_update(self, idx, eps, update):
        samples = []
        for i, e in enumerate(eps):
            experience = self._decompose(self.memory[range(*e)])# indexing will copy tensor out

            recalc = 0 == random.randint(0, self.recalc_delay)
            experience = update(
                    recalc,
                    [(j - e[0]).item() for j in idx[i]],
                    [self.allowed_mask[j] for j in range(*e)],
                    experience)
            #  experience = torch.cat(experience, 1)

#            experience = self.memory[range(*e)]
            if recalc: self.memory[range(*e)] = experience

            for j in idx[i]:
                assert j in range(*e)
                samples.append(experience[j - e[0]])
        return torch.stack(samples)

    def _decompose(self, samples):
        return [ # indexing inside len(idx) sized batch is cheaper than query chunk for len(idx) samples
                samples[:, sum(self.chunks[:i+1]):sum(self.chunks[:i+2])
            ] for i in range(len(self.chunks)-1) ]

    def _select(self, batch_size, back_view):
        assert len(self)
        idx = []
        bck = []
        size = 0
        start = 0 if not back_view else max(0, len(self) - back_view)
        while size < batch_size:
            i = random.randint(start, len(self) - 1)
            to_draw = random.randint(1, self.ep_draw)
            ep_idx = range(*self.back_pointers[i])
            selection = random.sample(ep_idx, min(len(ep_idx)-1, to_draw))

            selection = [s for s in selection if self.allowed_mask[s]]
            if not len(selection):
                continue

            idx.append(selection)
            size += len(idx[-1])
            bck.append(self.back_pointers[i])
        return idx, bck

    def _sync(self, e):
        cum_size = len(self)
        size = len(e)
        self_ptr = torch.tensor([[cum_size, cum_size + size]] * size)
        self.back_pointers = torch.cat([self.back_pointers, self_ptr]) if len(self.back_pointers) else self_ptr
        cum_size += size
        assert self.back_pointers[-1][1] == len(self.back_pointers)

    def _clip(self):
        if len(self) < self.capacity:
            return
        index = self.memory.shape[0] // 3
        new_bottom = self.back_pointers[index][1]
        self.memory = self.memory[new_bottom:]
        self.back_pointers = self.back_pointers[new_bottom:]
        self.back_pointers -= new_bottom
        self.allowed_mask = self.allowed_mask[new_bottom:]
        assert self.back_pointers[-1][1] == len(self.back_pointers)
        assert self.back_pointers[0][0] == 0
        assert len(self.allowed_mask) == len(self.back_pointers)
