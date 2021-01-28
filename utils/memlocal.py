import numpy as np
import random

from utils.fastmem import *
from timebudget import timebudget

class MemoryBoost:
    def __init__(self, descs, memory, credit_assign, brain, n_step, good_reach):
        self.fast_m = [ IRFastMemory( # TODO add branch for PER ( Priotized Experience Replay )
            desc, memory.chunks, memory.device) if cred.resampling else FastMemory(
                desc, memory.chunks, memory.device) for cred, desc in zip(credit_assign, descs) ]

        memory.device = 'cpu'
        self.memory = memory

        self.credit = credit_assign
        self.brain = brain
        self.n_step = n_step
        self.good_reach = good_reach

    def __len__(self):
        assert False

    def push(self, ep, chunks, e_i, goods):
        for i, fast_m in enumerate(self.fast_m):
            ir, episode = self._push(i, ep, chunks, e_i, goods)
            if not len(episode):
                return
            fast_m.push(episode, ir)

    def step(self, ind, desc):
        pass

    def sample(self, ind, desc):
        def update(ind): # curry curry
            def _update(recalc_feats, indices, allowed_mask, episode):
                return self._push_to_fast(recalc_feats, ind, indices, allowed_mask, episode)
            return _update

        def sample():
            with timebudget("FullMemory-sample"):
                return self.memory.sample(update(ind), desc.batch_size, desc.memory_size)

        if 0 == random.randint(0, desc.recalc_delay):
            sample()

        return self.fast_m[ind].sample

    @timebudget
    def _push_to_fast(self, recalc_feats, ind, indices, allowed_mask, episode):
        goals, states, memory, actions, probs, rewards, _, _, _, _, _ = episode

        _, ir, episode = self.credit[ind](goals, states, memory, actions, probs, rewards,
                self.brain, recalc=recalc_feats, indices=indices)

        idx = np.arange(len(episode))[allowed_mask]
        self.fast_m[ind].push(episode[idx], ir)

        return episode

    @timebudget
    def _push(self, ind, ep, chunks, e_i, goods):
        max_allowed = len(ep) - 1#self.n_step - 1
        allowed_mask = [ bool(sum(goods[
            max(0, i-self.good_reach):i+self.good_reach, e_i
            ])) for i in range(max_allowed)
                ] + [False] * (len(ep) - max_allowed)

        with timebudget("credit-assign"):
            _, ir, episode = self.credit[ind]( # it is double
                        *[ep[:, sum(chunks[:i+1]):sum(chunks[:i+2])] for i in range(len(chunks[:-1]))],
                        brain=self.brain,
                        recalc=True)

        idx = np.arange(len(episode))[allowed_mask]
        self.memory.push(episode, allowed_mask)
        return ir, episode[idx] # we returning copy not reference!
