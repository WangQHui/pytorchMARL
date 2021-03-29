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
                selector_nn = nn.Sequential(nn.Linear(conf.state_shape, conf.hyper_hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(conf.hyper_hidden_dim, conf.embed_dim))
                # 查询
                self.selector_extractors.append(selector_nn)
                # 加入qs
                if conf.nonlinear:
                    self.key_extractors.append(nn.Linear(conf.unit_dim+1, conf.mixing_embed_dim, bias=False))
                else:
                    self.key_extractors.append(nn.Linear(conf.unit_dim, conf.mixing_embed_dim, bias=False))
            if conf.weighted_head:
                self.hyper_w_head = nn.Sequential(nn.Linear(conf.state_shape, conf.hyper_hidden_dim),
                                                  nn.ReLU(),
                                                  nn.Linear(conf.hyper_hidden_dim, conf.n_head))
        else:
            for i in range(self.n_head):
                self.selector_extractors.append(nn.Linear(conf.state_shape, conf.mixing_embed_dim, bias=False))
                if conf.nonlinear:
                    self.key_extractors.append(nn.Linear(conf.unit_dim+1, conf.mixing_embed_dim, bias=False))
                else:
                    self.key_extractors.append(nn.Linear(conf.unit_dim, conf.mixing_embed_dim, bias=False))
            if conf.weighted_head:
                self.hyper_w_head = nn.Linear(conf.state_shape, conf.n_head)

        if conf.state_bias:
            self.V = nn.Sequential(nn.Linear(self.state_shape, conf.mixing_embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.mixing_embed_dim, 1))

    def forward(self, ):

