import torch
from utils.nes import *
from utils.policy import PPOHead

import random

class Actor(nn.Module): # decorator
    def __init__(self, net, action_size, f_mean_clip, f_scale_clip):
        super().__init__()
        self.net = net
        self.algo = PPOHead(action_size, f_scale_clip)
        self.f_mean_clip = f_mean_clip
    def forward(self, goal, state):
        state = torch.cat([goal, state], 1)
        x = self.net(state)
        x = self.f_mean_clip(x)
        return self.algo(x)

    def sample_noise(self, _):
        self.net.sample_noise(random.randint(0, len(self.net.layers) - 1))
    def remove_noise(self):
        self.net.remove_noise()

class ActorFactory: # proxy
    def __init__(self, layers, action_size, f_mean_clip, f_scale_clip, device):
        self.factory = NoisyNetFactory(layers, device)
        self.actor = lambda head: Actor(head, action_size, f_mean_clip, f_scale_clip)
    def head(self):
        return self.actor(self.factory.head())

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import numpy as np

def rl_ibounds(layer):
    b = 1. / np.sqrt(layer.weight.data.size(0))
    return (-b, +b)

def initialize_weights(layer):
    if type(layer) not in [nn.Linear, ]:
        return
#    nn.init.kaiming_uniform_(layer.weight) # does not appear working better ...
    nn.init.uniform_(layer.weight.data, *rl_ibounds(layer))

class Critic(nn.Module):
    def __init__(self, n_actors, n_rewards, state_size, action_size, fc1_units=400, fc2_units=300):
        super().__init__()

        state_size += 3

        action_size = action_size * 2 # action, mu, sigma

        #  state_size = 64
        self.fca = nn.Linear(action_size * n_actors, state_size)
        self.fcs = nn.Linear(state_size * n_actors, state_size * n_actors)

        self.fc1 = nn.Linear(state_size * (n_actors + 1), fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)#, bias=False)

        self.apply(initialize_weights)

        self.fc3 = nn.Linear(fc2_units, n_rewards)#, bias=False)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3) # seems this works better ? TODO : proper tests!!

    def forward(self, goals, states, actions):
        # process so actions can contribute to Q-function more effectively ( theory .. )
        actions = self.fca(actions[:, actions.shape[1]//3:]) # skip played action
        states = torch.cat([goals, states], 1)
        states = F.relu(self.fcs(states)) # push states trough as well
# after initial preprocessing let it flow trough main network in combined fashion
        xs = torch.cat((states, actions), dim=1)
        x = F.relu(self.fc1(xs))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class GANReward(nn.Module):
    def __init__(self, n_actors, n_rewards, state_size, goal_size, action_size, fc1_units=400, fc2_units=300):
        super().__init__()

        #  state_size = 64
        self.fcs = nn.Linear((goal_size + state_size + 2 * action_size) * n_actors, fc1_units)
        self.fc1 = nn.Linear(fc1_units, fc2_units)

        self.apply(initialize_weights)

        self.reward = nn.Sequential(
                nn.Linear(fc2_units, n_rewards),
                )#nn.ReLU())#nn.Softmax(dim=1))

        self.action = nn.Sequential(
                nn.Linear(fc2_units, action_size),
                )#nn.Tanh())

    def forward(self, goals, states, actions):
        actions = actions[:, actions.shape[1]//3:] # skip played action
        states = torch.cat([goals, states, actions], 1)
        x = F.relu(self.fcs(states))
        x = F.relu(self.fc1(x))
        return self.reward(x), self.action(x)
