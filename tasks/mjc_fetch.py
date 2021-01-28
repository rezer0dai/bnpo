import torch
import numpy as np
import random

#openai baselines for her, this is  reward_fun w/ comparsion to treshold
def goal_distance(goal_a, goal_b):
    return torch.norm(goal_a[:3] - goal_b[:3])

# lets use MROCS, and add to equation moving of object too
N_REWARDS = 2#3

def reward_norm(r, ddpg):
#    return (r+1)/10.
    if ddpg: return r#/10.
    else: return (r+1)/100.

import gym
def f_reward_make(env_name):
    fetch = gym.make(env_name)
    def f_reward(s, s2, n, goal, ddpg):
        # achieved goal vs target goal
#        r1 = fetch.compute_reward(s2[:len(goal)], goal, None)
        r1 = fetch.compute_reward(n[:len(goal)], goal, None)
#        r1 = -1 * (1e-3 < goal_distance(s2, goal)).double()
#        assert rx == r1, "{} vs {}".format(rx, r1)

#        r2 = fetch.compute_reward(
#                s2[:len(goal)], s[:len(goal)], None)
        # object is moving or not
        r2 = -1 * (goal_distance(s2, s) > 1e-3).double()
        r2 = -(r2+1) # reverse, if moving it is good == 0 else -1

#lets try to unify in favor of PPO
#        return np.array([
#                1e-3 if (0 == r1) else 0.,
#                -1e-3 if not(0 == r2 or 0 == r1) else 0.,
#                ])
        
        if not ddpg: return np.array([
                1e-3 if (0 == r1) else 0.,
                -1e-3 if not(0 == r2 or 0 == r1) else 0.,
                ])

        return np.array([
            -1e-1 if not(0 == r1) else 0.,
            -1e-1 if not(0 == r2 or 0 == r1) else 0.,
            ])

        if True:#"FetchReach-v1" == env_name:
# this may differe between simple fetch and others
            r2 = r2-1 # as in reacher moving is common
            # r2 means will emit signal all the time, PPO ?

        # grip to object ~ slide is bit of hook here ..

        if 0 == r1: r2 = r1 # all good
#        if 0 == r2: r3 = r2 # yep all good

        # kind of problematic for PPO :
        r1 = r1-1

        return np.array([
            reward_norm(r1, ddpg) / 10,
            reward_norm(r2, ddpg) / 1.,
#            reward_norm(r3, ddpg)
            ])

    return f_reward

#from tasks import Nice_plot
from tasks.oaiproc import GymGroup, GymRender

#@dataclass
class Info:
    def __init__(self, env_name, states, rewards, actions, custom_rewards, dones, goals, dist):
        self.states = states
        self.rewards = rewards
        self.actions = actions
        self.custom_rewards = custom_rewards
        self.dones = dones
        self.goals = goals

        self.dist = dist
        # filter eps where we achieve goal
#        self.goods = [0 == r for r in rewards]
        # filter eps where we touch object at least
#        if "FetchReach-v1" == env_name:
#            self.goods = [0 == r[1] for r in custom_rewards]
#        else:
#            self.goods = [0 < r[1] for r in custom_rewards]

        self.goods = [True] * len(rewards)

class StaticReacherProxy:
    def __init__(self, env_name, dock, n_env, prefix):
        self.ENV = GymGroup(env_name, dock, n_env, prefix)
        self.RENDER = GymRender(env_name, n_env)
        self.env_name = env_name
        self.learn_mode = False
        self.seeds = []
        self._reward_fun = f_reward_make(env_name)

    def reward_fun(self, states, states_1, n_states, goals, ddpg):
        rewards = np.concatenate([
            self._reward_fun(s, s2, n, g, ddpg
                ).reshape(1, -1) for s, s2, n, g in zip(
                    states, states_1, n_states, goals)])
#        print(rewards)
        return torch.tensor(rewards)
            
    def _state(self, 
            einfo, actions, 
            learn_mode=False, reset=False, seed=None):

        states, self.goals, rewards, dones = einfo

        # openai baselines
        states = torch.clamp(states, min=-200., max=200.)

        info = Info(self.env_name,
                states,
                torch.tensor(rewards).view(-1, 1),
                actions[:, :4],

                self.reward_fun( #we recalc for reward#2 for ep selection
                    self.info.states, states, states, self.goals, False
                    ) if self.info is not None else torch.tensor(
                        rewards).view(-1, 1).expand_as(
                    torch.zeros(len(rewards), N_REWARDS)),

                torch.tensor([1. if e else 0. for e in dones]).view(len(dones), -1).double(),
                self.goals,

                torch.stack([ goal_distance(s, g) for s, g in zip(states, self.goals) ]).view(-1, 1),
                )

        self.stats.append( torch.cat([
            info.states[:, :3] * 7,
            torch.tensor(rewards).view(-1, 1), 
            info.custom_rewards, info.actions, info.dist,
            info.goals * 7], 1) )

        self.info = info
        return info

    def _plot(self, agent):
        stats = np.asarray([s.numpy() for s in self.stats])
        best = max([sum(s) for s in stats.T[3]])
        bests = [i for i, s in enumerate(stats.T[3]) if sum(s) == best]
        longest = np.argmin([sum(s) for s in np.asarray(stats).T[-1, bests]])
        top = bests[longest]

        values = torch.stack([qa[top] for qa in agent.brain.qa_vs]).mean(1).numpy()
        future = torch.stack([qa[top] for qa in agent.brain.qa_fs]).mean(1).numpy()

        goals = stats[:-1, top, -3:]
        for i in range(3): # normalize
            trajectory = stats[:-1, top, i]
            emax, emin = trajectory.max(), trajectory.min()

            delta = (emax-emin) if i < 2 else (emin-emax)

            stats[:-1, top, i] = 2. * (trajectory - emin - delta / 2.) / 3.
            goals[:, i] = 2. * (goals[:, i] - emin - delta / 2.) / 3.

#        Nice_plot.plot_proxy(stats[:-1, top], goals, values, future)

    def reset(self, agent, seed, learn_mode):
        if learn_mode and not self.learn_mode:
            self._plot(agent)

        self.stats = []
        self.learn_mode = learn_mode

        if self.learn_mode:
            einfo = self.ENV.reset(seed)
        else:
            einfo = self.RENDER.reset(seed)

        self.info = None
        return self._state(
                einfo, 
                torch.zeros([len(einfo[0]), 4*3]),
                learn_mode, True, seed)

    def step(self, actions):
        if self.learn_mode:
            einfo = self.ENV.step(
                    actions[:, :actions.shape[1]//3].cpu().numpy())
        else:
            einfo = self.RENDER.step(
                    actions[:, :actions.shape[1]//3].cpu().numpy())
        return self._state(einfo, actions)
