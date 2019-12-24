import random
import numpy as np
import torch

import utils.policy as policy

class CreditAssignment:# temporary resampling=False, if proved good then no default, need to choose!
    def __init__(self, cind, gae, n_step, floating_step, gamma, gae_tau, resampling=False, kstep_ir=False, clip=None):
        self.cind = cind
        self.gae = gae
        self.n_step = n_step
        self.floating_step = floating_step
        self.resampling = resampling
        self.kstep_ir = kstep_ir
        self.clip = clip
        self.policy = policy.GAE(gamma, gae_tau) if gae else policy.KSTEP(gamma)

    def __call__(self,
            goals, states, features, actions, probs, orig_rewards,
            brain, recalc, indices=[]):

        her, indices, n_steps, n_indices = self._random_n_step(
                len(states), indices, recalc)

        #  if her: print((n_indices[:-self.n_step]-np.arange(len(n_indices)-self.n_step)).mean(), sum(her_step_inds), n_indices)

# even if we dont recalc here, HER or another REWARD 'shaper' will do its job!!
        ( rewards, goals, states, n_goals, n_states ) = self._update_goal(her,
            orig_rewards,
            goals, states,
            states[list(range(1, len(states))) + [-1]],
            goals[n_indices], states[n_indices],
            actions,
            indices,
            n_steps)

        if recalc or self.resampling:#we need everytime to resample! TODO: make it optional ?
            features, ir_ratio = brain.recalc_feats(goals, states, actions, probs, n_steps,
                    self.resampling, self.kstep_ir, self.cind, self.clip)
        else:
            ir_ratio = torch.ones(len(features))
#        elif self.gae:
#            n_steps[ [i for i in range(len(n_steps)) if i not in indices] ] = 0

        c, d = self._redistribute_rewards( # here update_goal worked on its own form of n_goal, so dont touch it here!
                brain, n_steps, rewards, goals, states, features, actions, stochastic=True)

        # we by defaulf skip
        assert c.shape == rewards.shape
        assert not self.resampling or ir_ratio is not None
        return her, ir_ratio, torch.cat([
                    goals, states, features, actions, probs, orig_rewards,
                    n_goals, n_states, features[n_indices],
                    c, d.view(-1, 1)] , 1)

    def _update_goal(self, her, rewards, goals, states, states_1, n_goals, n_states, actions, her_step_inds, n_steps):
        if not her:
            return ( rewards, goals, states, n_goals, n_states )
        return self.update_goal(rewards, goals, states, states_1, n_goals, n_states, actions, her_step_inds, n_steps)

    # duck typing
    def update_goal(self, rewards, goals, states, states_1, n_goals, n_states, actions, her_step_inds, n_steps):
        assert False

    def _assign_credit(self, brain, n_steps, rewards, goals, states, features, actions, stochastic):
        if not self.gae: return self.policy(n_steps, rewards)
        else: return self.policy(
                n_steps,
                rewards[:len(states) - 1],
                brain.qa_future(goals, states, features, actions, self.cind))

    def _redistribute_rewards(self, brain, n_steps, rewards, goals, states, features, actions, stochastic):
        # n-step, n-discount, n-return - Q(last state)
        discounts, credits = self._assign_credit(
                brain, n_steps, rewards, goals, states, features, actions, stochastic)

        discounts = torch.tensor([discounts + self.n_step*[0]])
        credits = torch.cat([
            torch.stack(credits), torch.zeros(self.n_step, len(credits[0]))])

        return ( credits, discounts )

    def _do_n_step(self, i):
        return self.n_step if not self.floating_step else random.randint(1, self.n_step)

    def _random_n_step(self, length, _indices, _recalc):
        return (False, None, *self._do_random_n_step(length, self._do_n_step))

    def _do_random_n_step(self, length, n_step):
        # + with indices you want to skip last self.n_step!!
        n_steps = np.array([ n_step(i) for i in range(length - self.n_step) ])
        #  n_steps[-1] = min(self.n_step-1, n_steps[-1])
        indices = n_steps + np.arange(len(n_steps))
        indices = np.hstack([indices, self.n_step * [-1]])
        return (n_steps, indices)
