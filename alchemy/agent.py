import torch
import numpy as np

from alchemy.brain import Brain
from utils.memory import Memory
from utils.rl_algos import BrainOptimizer

import time, random

from timebudget import timebudget

class BrainDescription:
    def __init__(self,
            memory_size, batch_size,
            optim_pool_size, optim_epochs, optim_batch_size, recalc_delay,
            lr_actor, learning_delay, learning_repeat,
            sync_delta_a, sync_delta_c, tau_actor, tau_critic,
            bellman, ppo_eps, natural, mean_only, separate_actors, prio_schedule=None
            ):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.optim_epochs = optim_epochs
        self.optim_batch_size = optim_batch_size
        self.optim_pool_size = optim_pool_size
        self.recalc_delay = recalc_delay
        self.sync_delta_a = sync_delta_a
        self.sync_delta_c = sync_delta_c
        self.learning_delay = learning_delay
        self.learning_repeat = learning_repeat
        self.lr_actor = lr_actor
        self.tau_actor = tau_actor
        self.tau_critic = tau_critic
        self.bellman = bellman
        self.ppo_eps = ppo_eps
        self.natural = natural
        self.mean_only = mean_only
        self.separate_actors = separate_actors

        self.prio_schedule = prio_schedule

        self.counter = 0

    def __repr__(self):
        return str([
            self.memory_size, "<- memory_size;",
            self.batch_size, "<- batch_size;",
            self.optim_epochs, "<- optim_epochs;",
            self.optim_batch_size, "<- optim_batch_size;",
            self.optim_pool_size, "<- optim_pool_size;",
            self.recalc_delay, "<- recalc_delay;",
            self.sync_delta_a, "<- sync_delta_a;",
            self.sync_delta_c, "<- sync_delta_c;",
            self.learning_delay, "<- learning_delay;",
            self.learning_repeat, "<- learning_repeat;",
            self.lr_actor, "<- lr_actor;",
            self.tau_actor, "<- tau_actor;",
            self.tau_critic, "<- tau_critic;",
            self.ppo_eps, "<- ppo_eps;",
            self.natural, "<- natural;",
            self.mean_only, "<- mean_only;",
            ])

class Agent:
    def __init__(self, device,
            brains, experience,
            Actor, Critic, goal_encoder, encoder,
            n_agents, detach_actors, detach_critics, stable_probs,
            resample_delay, min_step,
            state_size, action_size,
            freeze_delta, freeze_count,
            lr_critic, clip_norm,
            model_path, save, load, delay,
            ):

        self.device = device
        self.freeze_delta = freeze_delta
        self.freeze_count = freeze_count

        n_actors = 1 if not detach_actors else len(brains)
        n_critics = 1 if not detach_critics else len(brains)
        self.n_targets = max(n_actors, n_critics)

        self.brain = Brain(device,
            Actor, Critic, encoder, goal_encoder,
            n_agents, n_actors, n_critics, stable_probs,
            resample_delay,
            lr_critic, clip_norm,
            model_path=model_path, save=save, load=load, delay=delay
            )
        self.brain.share_memory() # at this point we dont know if this is needed

        self.bd_desc = brains
        self.algo = [ BrainOptimizer(self.brain, desc) for desc in self.bd_desc ]

        self.counter = 0
        self.freeze_d = 0
        self.freeze_c = 0
        self.exps = experience(self.bd_desc, self.brain)

        self.min_step = min_step

    def step(self, step):
        for i, desc in enumerate(self.bd_desc):
            self.exps.step(i, desc)

        self._clocked_step()

    @timebudget
    def _clocked_step(self):
        for a_i, bd in self._select_algo():
            for _ in range(bd.learning_repeat // bd.optim_epochs):
                bd.counter += 1
                self._encoder_freeze_schedule()

                with timebudget("learn-round"):
                    batch = self.exps.sample(a_i, bd)
                    self.brain.learn(
                            batch,
                            0 if bd.counter % bd.sync_delta_a else bd.tau_actor,
                            0 if bd.counter % bd.sync_delta_c else bd.tau_critic,
                            self.algo[a_i], a_i, bd.mean_only, bd.separate_actors)

    def save(self, goals, states, memory, actions, probs, rewards, goods, finished):
        episode_batch = (goals, states, memory, actions, probs, rewards)
        goods = np.asarray(goods)
        if len(goods) < self.min_step:
            return
        #  goals, states, memory, actions, probs, rewards
        chunks = [0] + [e[0].shape[1] for e in episode_batch]

        full_batch = []
        for i in range(len(episode_batch[0])):
            data = torch.cat([chunk[i] for chunk in episode_batch], 1)
            full_batch.append(data)
        full_batch = torch.stack(full_batch).transpose(0, 1).contiguous()

        for e_i, ep in enumerate(full_batch):
            if not sum(goods[:, e_i]):
                continue
            self.exps.push(ep, chunks, e_i, goods)

    def _select_algo(self):
        self.counter += 1
        for i, bd in enumerate(self.bd_desc):
            if self.counter % bd.learning_delay:
                continue
#            print("--->", len(bd.fast_m), len(self.exps))
#            if len(self.exps) < bd.optim_batch_size:
#                continue # out of process for memserver !!
            if len(self.exps.fast_m[i]) < bd.optim_batch_size:
                continue
            yield i, bd

    def _encoder_freeze_schedule(self):
        if not self.freeze_delta:
            return
        self.freeze_d += (0 == self.freeze_c)
        if self.freeze_d % self.freeze_delta:
            return
        if not self.freeze_c:
            self.brain.freeze_encoders()
        self.freeze_c += 1
        if self.freeze_c <= self.freeze_count:
            return
        self.freeze_c = 0
        self.brain.unfreeze_encoders()

    def exploit(self, goal, state, history, tind):
        return self.brain.exploit(goal.to(self.device), state.to(self.device), history.to(self.device), tind)

    def explore(self, goal, state, history, t):
        return self.brain.explore(goal.to(self.device), state.to(self.device), history.to(self.device), t)

    def sync_target(self, b, blacklist):
        self.brain.sync_target(b, blacklist)

    def sync_explorer(self, b, blacklist):
        self.brain.sync_explorer(b, blacklist)
