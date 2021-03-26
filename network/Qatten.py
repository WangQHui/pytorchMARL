import torch.nn as nn
import torch
import torch.nn.functional as F
import  numpy as np

class Qatten(nn.Module):
    def __init__(self, conf):
        super(Qatten, self).__init__()

        self.conf = conf
        #self.n_agents = conf.n_agents
        #self.n_actions = conf.n_actions

        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()

        if conf.two_hyper_layers:
            # 多头注意力机制
            for i in range(self.n_head):
                selector_nn = nn.Sequential(nn.Linear(self.state_shape, self.hyper_hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(self.hyper_hidden_dim, conf.state_shape+conf.n_agents*conf.n_actions))
                # 查询
                self.selector_extractors.append(selector_nn)
                # 加入qs
                if conf.nonlinear:
                    self.key_extractors.append(nn.Linear(self.state_shape+1, self.))
