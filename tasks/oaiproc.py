from torch.multiprocessing import Queue, Process
import gym

import torch
import numpy as np

class GymProxy(Process):
    def __init__(self, env_name, dock, port, prefix, ind):
        super().__init__()
        self.query = Queue()
        self.data = Queue()

        self.port = port
        self.env_name = env_name

        self.name = "%s_gym_dock_%i"%(prefix, ind)

    def run(self):
        self.cmd = { 
                "create" : self._create,
                "step" : self._step,
                "reset" : self._reset
                }

        while True: # single thread is fine
            data = self.query.get()
            cmd, data = data
            data = self.cmd[cmd](data)
            self.data.put(data)

    def _create(self, data):
        print(self.name, "create", data)
        self.env = gym.make(data)
        return self._reset(0)

    def _reset(self, seed):
#        print(self.name, "reset /w seed ", seed)
        if seed: self.env.seed(seed)
        return (self.env.reset(), 0, False, None)

    def _step(self, data):
        return self.env.step(data)

# bit overkill everytime new sock, but perf seems fast anyway
    def _do(self, action, data, asnc):
        self.query.put((action, data))
        if asnc: return
        return self.data.get()

    def make(self):
        return self._do("create", self.env_name, False)

    def reset(self, seed):
        return self._do("reset", seed, False)

    def act(self, actions):
        return self._do("step", actions, True)

    def step(self):
        return self.data.get()

class GymGroup:
    def __init__(self, env_name, dock, n_env, prefix):
        self.gyms = [
                GymProxy(env_name, dock, 5001, prefix, i
                    ) for i in range(n_env)]
        for gym in self.gyms:
            gym.start()
        for gym in self.gyms:
            gym.make()

    def reset(self, seeds):
        obs = np.concatenate([
            self._process(
                gym.reset(int(seed))
                ) for gym, seed in zip(self.gyms, seeds) ], 0)
        return self._decouple(obs)

    def step(self, actions):
        a_s = len(actions) // len(self.gyms)

        for i, gym in enumerate(self.gyms):
            gym.act(actions[i])
                
        obs = np.concatenate([
            self._process(
                gym.step()
                ) for gym in self.gyms ], 0)
        return self._decouple(obs)

    def _step(self, data):
        return self.env.step(data)

# bit overkill everytime new sock, but perf seems fast anyway
    def _do(self, action, data, asnc):
        self.query.put((action, data))
        if asnc: return
        return self.data.get()

    def make(self):
        return self._do("create", self.env_name, False)

    def reset(self, seed):
        return self._do("reset", seed, False)

    def act(self, actions):
        return self._do("step", actions, True)

    def step(self):
        return self.data.get()

class GymGroup:
    def __init__(self, env_name, dock, n_env, prefix):
        self.gyms = [
                GymProxy(env_name, dock, 5001, prefix, i
                    ) for i in range(n_env)]
        for gym in self.gyms:
            gym.start()
        for gym in self.gyms:
            gym.make()

    def reset(self, seeds):
        obs = np.concatenate([
            self._process(
                gym.reset(int(seed))
                ) for gym, seed in zip(self.gyms, seeds) ], 0)
        return self._decouple(obs)

    def step(self, actions):
        a_s = len(actions) // len(self.gyms)

        for i, gym in enumerate(self.gyms):
            gym.act(actions[i])
                
        obs = np.concatenate([
            self._process(
                gym.step()
                ) for gym in self.gyms ], 0)
        return self._decouple(obs)

    def _process(self, data):
        obs, reward, done, info = data
        return np.concatenate([
            obs["achieved_goal"], obs["observation"], obs["desired_goal"],
            [reward], [done]]).reshape(1, -1)

    def _decouple(self, obs):
        return (
                torch.from_numpy(obs[:, :-3-2]), 
                torch.from_numpy(obs[:, -3-2:-2]),
                obs[:, -2], obs[:, -1])

import gym

class GymRender:
    def __init__(self, env_name, n_env):
        self.n_env = n_env
        self.env = gym.make(env_name)

    def reset(self, seed):
        self.env.seed(int(seed[0]))
        obs = self.env.reset()
        state = torch.from_numpy(
                    np.concatenate([obs['achieved_goal'], obs['observation']])).expand_as(
                torch.ones(self.n_env, len(obs['observation']) + len(obs['achieved_goal'])))
        goals = torch.from_numpy(obs['desired_goal']).expand_as(
                torch.ones(self.n_env, len(obs['desired_goal'])))
        return (
                state, goals, 
                np.zeros([len(state), 1]), #rewards
                np.zeros([len(state), 1])) #dones

    def step(self, actions):
        obs, r, d, i = self.env.step(actions[0])

#        self.env.render() # whole purpose...

        state = torch.from_numpy(#obs['observation']).expand_as(
                    np.concatenate([obs['achieved_goal'], obs['observation']])).expand_as(
                torch.ones(self.n_env, len(obs['observation']) + len(obs['achieved_goal'])))
        goals = torch.from_numpy(obs['desired_goal']).expand_as(
                torch.ones(self.n_env, len(obs['desired_goal'])))
        return (
                state, goals, 
                np.zeros([len(state), 1]) + r, #rewards
                np.zeros([len(state), 1]) + d) #dones

