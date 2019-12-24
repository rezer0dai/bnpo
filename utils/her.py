import random
import numpy as np
from utils.credit import CreditAssignment

class HER(CreditAssignment):
    def __init__(self, cind, her_delay, gae, n_step, floating_step, gamma, gae_tau, resampling=False, kstep_ir=False, clip=None):
        super().__init__(cind, gae, n_step, floating_step, gamma, gae_tau, resampling, kstep_ir, clip)
        self.her_delay = her_delay

    def _random_n_step(self, length, indices, recalc):
        if recalc or random.randint(0, self.her_delay):
            #  print(recalc, len(indices), random.randint(0, self.her_delay))
            return super()._random_n_step(length, None, recalc)

        her_step_inds = self._her_indices(length, indices)
        n_step = lambda i: 1 if her_step_inds[i] else self._do_n_step(1)
        return (True, her_step_inds, *self._do_random_n_step(length, n_step))

    def _her_indices(self, ep_len, inds):
        inds = sorted(inds)
        cache = np.zeros(ep_len)
        cache[ self._her_select_idx(inds) ] = 1
        return cache

    def _her_select_idx(self, inds):
        collision_free = lambda i, ind: ind-self.n_step>inds[i-1] and ind+1!=inds[i+1]
        hers = [-1] + [ i for i, ind in enumerate(inds[1:-1]) if collision_free(i+1, ind) ]

        pivot = 1
        indices = [ inds[0] ]
        hers.append(len(inds)-1)

        her_draw = lambda i: 0 != random.randint(0, 1 + (i - hers[pivot-1]))

        for i, ind in enumerate(inds[1:]):
            if i == hers[pivot] or indices[-1]+1==ind or her_draw(i) and indices[-1] == inds[i]:
                indices.append(ind)
            if i == hers[pivot]:
                pivot += 1
        return indices
