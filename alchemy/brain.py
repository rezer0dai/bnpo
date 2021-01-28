import numpy as np
import random, copy, sys

from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim

torch.set_default_tensor_type('torch.DoubleTensor')

from utils.ac import *

from utils.polyak import POLYAK as META
#from utils.foml import FOML as META
#from utils.reptile import REPTILE as META

from timebudget import timebudget

class Brain(META):
    def __init__(self,
            device,
            Actor, Critic, encoder, goal_encoder,
            n_agents, n_actors, n_critics, stable_probs,
            resample_delay,
            lr_critic, clip_norm,
            model_path, save, load, delay
            ):
        super().__init__(model_path, save, load, delay)

        self.clip_norm = clip_norm
        self.resample_delay = resample_delay

        self.losses = []

        self.n_actors = n_actors
        self.stable_probs = stable_probs

        encoder.share_memory()
        goal_encoder.share_memory()

        Path(model_path).mkdir(parents=True, exist_ok=True)

        nes = Actor()
        self.ac_explorer = ActorCritic(encoder, goal_encoder,
                    [ nes.head() ],
                    [ Critic() for _ in range(n_critics) ], n_agents).to(device)

        self.ac_target = ActorCritic(encoder, goal_encoder,
                    [ Actor().head() for _ in range(n_actors) ],
                    [ Critic() for _ in range(n_critics) ], n_agents).to(device)

        print(self.ac_target)
        print(self.ac_explorer)
        # sync
        for target in self.ac_target.actor:
            self.polyak_update(self.ac_explorer.actor[0].parameters(), target, 1.)

        for i in range(n_critics):
            self.polyak_update(self.ac_target.critic[i].parameters(), self.ac_explorer.critic[i], 1.)

        #  self.init_meta(lr=1e-3)

        self.load_models(0, "eac")
        self.save_models_ex(0, "eac")

        self.critic_optimizer = [ optim.Adam(
            self.ac_explorer.critic_parameters(i), lr=lr_critic) for i in range(n_critics) ]

        self.resample(0)

        self.ag = []
        self.qa_vs = []
        self.qa_fs = []

    @timebudget
    def learn(self, batches, tau_actor, tau_critic, backward_policy, tind, mean_only, separate_actors):
        for batch in batches():
            self._learn(batch, tau_actor, tau_critic, backward_policy, tind, mean_only, separate_actors)

    @timebudget
    def _learn(self, batch, tau_actor, tau_critic, backward_policy, tind, mean_only, separate_actors):
        w_is, (goals, states, memory, actions, probs, _, n_goals, n_states, n_memory, n_rewards, n_discounts) = batch

        if not len(goals):
            return
        assert len(goals)

        self.losses.append([])

#        print("LEARN!", len(goals))
# resolve indexes
        if separate_actors:
            i = tind % len(self.ac_target.actor)
        else:
            i = random.randint(0, len(self.ac_target.actor)-1)

        a_i = tind % len(self.ac_explorer.actor)
        cind = tind % len(self.ac_target.critic)

# SELF-play ~ get baseline
        with torch.no_grad():
            n_qa, _ = self.ac_target(n_goals, n_states, n_memory, cind, i, mean_only)
        # TD(0) with k-step estimators
        td_targets = n_rewards + n_discounts * n_qa

# activate gradients ~ SELF-play
        qa, dists = self.ac_explorer(goals, states, memory, cind, a_i, mean_only)

# NATURAL GRADIENT section
        def surrogate_loss():
            _qa, _dists = self.ac_explorer(goals, states, memory, cind, a_i)
            #  return -((-_dists.log_prob(actions)+dists.log_prob(actions)).mean(1) * (qa - td_targets).mean(1)).mean()
            #NDDPG
            return -((-_dists.log_prob(actions)+dists.log_prob(actions)).mean(1) * (_qa - td_targets).mean(1) * w_is).mean()
            #PPO
            return -((_dists.log_prob(actions)-dists.log_prob(actions)).mean(1)*(td_targets - qa).mean(1)).mean()

# learn ACTOR ~ explorer
        pi_loss = backward_policy(
                qa, td_targets, w_is,
                probs, actions, dists,
                surrogate_loss)

#        cind = cind ^ 1 # lets try this, train Q by other agent, lets own stable
#        with torch.no_grad():
#            n_qa, _ = self.ac_target(n_goals, n_states, n_memory, cind, i ^ 1, mean_only)
#        td_targets = n_rewards + n_discounts * n_qa

# learn CRITIC ~ explorer + target
        # estimate reward
        q_replay = self.ac_explorer.value(goals, states, memory, actions, cind)
        # calculate loss via TD-learning
#        critic_loss = F.mse_loss(q_replay, td_targets) * w_is.mean()
        critic_loss = ((q_replay - td_targets).pow(2).mean(1) * w_is).mean()
        # learn!
        self.backprop(self.critic_optimizer[cind], critic_loss, self.ac_explorer.critic_parameters(cind))

        # propagate updates to target network ( network we trying to effectively learn )
        self.meta_update(
                cind,
                self.ac_explorer.critic[cind].parameters(),
                self.ac_target.critic[cind],
                tau_critic)
        # DEBUG
        self.losses[-1].append(critic_loss.item())

        # propagate updates to actor target network ( network we trying to effectively learn )
        self.meta_update(
                tind,
                self.ac_explorer.actor[a_i].parameters(),
                self.ac_target.actor[tind % len(self.ac_target.actor)],
                tau_actor)

        self.save_models(0, "eac")

        #  for target in self.ac_target.actor:
        #      target.remove_noise()

        self.losses[-1] = [pi_loss.item()]+self.losses[-1]

    def resample(self, t):
        if 0 != t % self.resample_delay:
            return
        for actor in self.ac_explorer.actor:
            actor.sample_noise(t // self.resample_delay)

    def explore(self, goal, state, memory, t): # exploration action
        self.ag = []
        self.qa_vs = []
        self.qa_fs = []

        a_i = random.randint(0, len(self.ac_target.actor) - 1)
        self.resample(t)
        with torch.no_grad(): # should run trough all explorers i guess, random one to choose ?
            e_dist, mem = self.ac_explorer.act(goal, state, memory, 0)

            if not self.stable_probs:
                t_dist = e_dist
            else:
                t_dist, _ = self.ac_target.act(goal, state, memory, a_i)

        return e_dist, mem.cpu(), t_dist

    def exploit(self, goal, state, memory, tind): # exploitation action
        with torch.no_grad():
            dist, mem = self.ac_target.act(goal, state, memory, tind % len(self.ac_target.actor))

            self.ag.append(dist.sample())
            self.qa_vs.append(self.ac_explorer.value(goal, state, memory, dist.params(True), tind % len(self.ac_explorer.critic)))
            self.qa_fs.append(self.ac_target.value(goal, state, memory, dist.params(True), tind % len(self.ac_target.critic)))

        return dist, mem.cpu(), dist

    def qa_future(self, goals, states, memory, actions, cind):
        with torch.no_grad():
            return self.ac_target.value(goals, states, memory, actions, cind % len(self.ac_target.critic)).cpu()

    @timebudget
    def backprop(self, optim, loss, params, callback=None, just_grads=False):
        # learn
        optim.zero_grad() # scatter previous optimizer leftovers
        loss.backward(retain_graph=callback is not ()) # propagate gradients
        torch.nn.utils.clip_grad_norm_(params, self.clip_norm) # avoid (inf, nan) stuffs

#        for p in params:
#            if not torch.isnan(p.detach()).sum(): continue
#            print("NaN IN PARAMS!!", callback is None)
#            return

        if just_grads:
            return # we want to have active grads but not to do backprop!

        if callback is not None:
            optim.step(callback) # trigger backprop with natural gradient
        else:
            optim.step() # trigger backprop

    @timebudget
    def recalc_feats(self, goals, states, actions, e_log_probs, n_steps, resampling, kstep_ir, cind, clip):
        with torch.no_grad():
            _, f = self.ac_target.encoder.extract_features(states)

            if not resampling:
                return f, torch.ones(len(f))

            if not self.stable_probs:
                e_dist, _ = self.ac_explorer.act(goals, states, f, 0)
                e_log_probs = e_dist.log_prob(actions)

            t_dist, _ = self.ac_target.act(goals, states, f, cind)
            t_log_probs = t_dist.log_prob(actions)

            ir_ratio = (t_log_probs - e_log_probs).exp().mean(1)

            if kstep_ir:
                ir_ratio = torch.tensor([ir_ratio[i:i+k].sum() for i, k in enumerate(n_steps)])

            ir_ratio = torch.clamp(ir_ratio, min=-clip, max=clip)
        return f, ir_ratio

    def freeze_encoders(self):
        self.ac_explorer.freeze_encoders()
        self.ac_target.freeze_encoders()

    def unfreeze_encoders(self):
        self.ac_explorer.unfreeze_encoders()
        self.ac_target.unfreeze_encoders()
