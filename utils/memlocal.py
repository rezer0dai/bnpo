import numpy as np
import random

from utils.fastmem import FastMemory
from timebudget import timebudget

class MemoryBoost:
    def __init__(self, descs, memory, credit_assign, brain, n_step, good_reach):
        self.fast_m = [ FastMemory(
            desc, memory.chunks, memory.device) for desc in descs ]

        memory.device = 'cpu'
        self.memory = memory

        self.credit = credit_assign
        self.brain = brain
        self.n_step = n_step
        self.good_reach = good_reach

    def __len__(self):
        assert False

    def push(self, ep, chunks, e_i, goods):
        episode = self._push(ep, chunks, e_i, goods)
        for fast_m in self.fast_m:
            fast_m.push(episode)

    def step(self, ind, desc):
        pass

    def sample(self, ind, desc):
        def update(ind): # curry curry
            def _update(recalc, indices, allowed_mask, episode):
                return self._push_to_fast(ind, recalc, indices, allowed_mask, episode)
            return _update

        def sample():
            with timebudget("FullMemory-sample"):
                yield self.memory.sample(update(ind), desc.batch_size, desc.memory_size)

        if random.randint(0, desc.recalc_delay):
            return self.fast_m[ind].sample
        else:
            return sample

    @timebudget
    def _push_to_fast(self, ind, recalc, indices, allowed_mask, episode):
        goals, states, memory, actions, probs, rewards, _, _, _, _, _ = episode

        _, episode = self.credit[ind](goals, states, memory, actions, probs, rewards,
                self.brain, recalc=recalc, indices=indices)

        idx = np.arange(len(episode))[allowed_mask]
        self.fast_m[ind].push(episode[idx])

        return episode

    @timebudget
    def _push(self, ep, chunks, e_i, goods):
        max_allowed = len(ep) - self.n_step - 1
        allowed_mask = [ bool(sum(goods[i:i+self.good_reach, e_i])) for i in range(max_allowed)
                ] + [False] * (len(ep) - max_allowed)

        with timebudget("credit-assign"):
            _, episode = self.credit[-1]( # it is double
                        *[ep[:, sum(chunks[:i+1]):sum(chunks[:i+2])] for i in range(len(chunks[:-1]))],
                        brain=self.brain,
                        recalc=True)

        idx = np.arange(len(episode))[allowed_mask]
        self.memory.push(episode, allowed_mask)
        return episode[idx] # we returning copy not reference!
