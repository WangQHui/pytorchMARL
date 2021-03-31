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

        # 用V代替最后一层的偏移量
        self.V = nn.Sequential(nn.Linear(self.state_shape, conf.mixing_embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.mixing_embed_dim, 1))

    def forward(self, agent_qs, states, actions):
        states = states.reshape(-1, self.state_shape)
        # 从全局状态获得agent自己的特征
        unit_states = states[:, :self.unit_dim*self.n_agents]
        unit_states = unit_states.reshape(-1, self.n_agents, self.unit_dim)
        unit_states = unit_states.permute(1, 0, 2)

        # agents_qs:(batch_size, 1, agent_num)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        if self.conf.nonlinear:
            unit_states = torch.cat((unit_states, agent_qs.permute(2, 0, 1)), dim=2)
            # states:(batch_size, state_shape)
            all_head_selectors = [sel_ext(states) for sel_ext in self.selector_extractors]
            # all_head_selectors:(head_num, batch_size, mixing_embed_dim)
            # unit_states:(agent_num,batch_size,unit_dim)
            all_head_keys = [[k_ext(enc) for enc in unit_states] for k_ext in self.key_extractors]
            # all_head_keys:(head_num, agent_num, embed_dim)

            # calculate attention per head
            head_attend_logits = []
            head_attend_weights = []
            for curr_head_keys, curr_head_selector in zip(all_head_keys, all_head_selectors):
                # curr_head_keys:(agent_num, batch_size, embed_dim)
                # curr_head_electors:(batch_size, embed_dim)
                # (batch_size, 1, embed_dim) * (batch_size, embed_dim, agent_num)
                attend_logits = torch.matmul(curr_head_selector.view(-1, 1, self.mixing_embed_dim),
                                             torch.stack(curr_head_keys).permute(1, 2, 0))
                # attend_logits:(batch_size, 1, agent_num)
                # scale dot-products 


