import numpy as np
import torch
from torch.distributions import one_hot_categorical

class RolloutWorker:
    def __init__(self, env, agents, conf):
        self.env = env
        self.agents = agents
        self.episode_limit = conf.episode_limit
        self.n_agents = conf.n_agents
        self.n_actions = conf.n_actions
        self.obs_state = conf.obs_state
        self.conf = conf

        self.epsilon = conf.epsilon
        self.anneal_epsilon = conf.anneal_epsilon
        self.end_epsilon = conf.end_epsilon
        print('Rollout Worker inited')

    def generated_episode(self, episode_num=None, evaluated=False):
