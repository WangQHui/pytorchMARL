import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class Qatten(nn.Module):
    def __init__(self, args):
        super(Qatten, self).__init__()

        self.args = args

        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()

        if args.two_hyper_layers:
            # 多头注意力机制
            for i in range(self.n_head):
                selector_nn = nn.Sequential(nn.Linear(args.state_shape, args.hyper_hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(args.hyper_hidden_dim, args.mixing_embed_dim))
                # 查询
                self.selector_extractors.append(selector_nn)
                # 加入qs
                if args.nonlinear:
                    self.key_extractors.append(nn.Linear(args.unit_dim+1, args.mixing_embed_dim, bias=False))
                else:
                    self.key_extractors.append(nn.Linear(args.unit_dim, args.mixing_embed_dim, bias=False))
            if args.weighted_head:
                self.hyper_w_head = nn.Sequential(nn.Linear(args.state_shape, args.hyper_hidden_dim),
                                                  nn.ReLU(),
                                                  nn.Linear(args.hyper_hidden_dim, args.n_head))
        else:
            for i in range(self.n_head):
                self.selector_extractors.append(nn.Linear(args.state_shape, args.mixing_embed_dim, bias=False))
                if args.nonlinear:
                    self.key_extractors.append(nn.Linear(args.unit_dim+1, args.mixing_embed_dim, bias=False))
                else:
                    self.key_extractors.append(nn.Linear(args.unit_dim, args.mixing_embed_dim, bias=False))
            if args.weighted_head:
                self.hyper_w_head = nn.Linear(args.state_shape, args.n_head)

        # 用V代替最后一层的偏移量
        self.V = nn.Sequential(nn.Linear(self.state_shape, args.mixing_embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.mixing_embed_dim, 1))

    def forward(self, agent_qs, states, actions):
        states = states.reshape(-1, self.state_shape)
        # 从全局状态获得agent自己的特征
        unit_states = states[:, :self.unit_dim*self.n_agents]
        unit_states = unit_states.reshape(-1, self.n_agents, self.unit_dim)
        unit_states = unit_states.permute(1, 0, 2)  # unit_states:(agent_num,batch_size,unit_dim)

        # agents_qs:(batch_size, 1, agent_num)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        if self.conf.nonlinear:
            unit_states = torch.cat((unit_states, agent_qs.permute(2, 0, 1)), dim=2)
            # states:(batch_size, state_shape)
            all_head_selectors = [sel_ext(states) for sel_ext in self.selector_extractors]
            # all_head_selectors:(head_num, batch_size, mixing_embed_dim)
            # unit_states:(agent_num,batch_size,unit_dim)
            all_head_keys = [[k_ext(enc) for enc in unit_states] for k_ext in self.key_extractors]
            # all_head_keys:(head_num, agent_num, mixing_embed_dim)

            # calculate attention per head
            head_attend_logits = []
            head_attend_weights = []
            for curr_head_keys, curr_head_selector in zip(all_head_keys, all_head_selectors):
                # curr_head_keys:(agent_num, batch_size, mixing_embed_dim)
                # curr_head_electors:(batch_size, mixing_embed_dim)
                # (batch_size, 1, embed_dim) * (batch_size, embed_dim, agent_num)
                attend_logits = torch.matmul(curr_head_selector.view(-1, 1, self.mixing_embed_dim),
                                             torch.stack(curr_head_keys).permute(1, 2, 0))
                # attend_logits:(batch_size, 1, agent_num)
                # scale dot-products by size of key (from Attention is All You Need)
                scaled_attend_logits = attend_logits / np.sqrt(self.mixing_embed_dim)
                if self.conf.mask_dead:
                    # actions:(episode_batch, episode_len - 1, agent_num, 1)
                    actions = actions.reshape(-1, 1, self.n_agents)
                    # actions:(batch_size, 1, agent_num)
                    scaled_attend_logits[actions == 0] = - 99999999  # action == 0 意味着 unit is dead
                attend_weights = F.softmax(scaled_attend_logits, dim=2)  # (batch_size, 1, agent_num)

                head_attend_logits.append(attend_logits)
                head_attend_weights.append(attend_weights)

            head_attend = torch.stack(head_attend_weights, dim=1)  # (batch_size, self.n_head, self.n_agents)
            head_attend = head_attend.view(-1, self.n_head, self.n_agents)

            v = self.V(states).view(-1, 1)  # v:(bs, 1)
            # head_qs:[head_num, bs, 1]
            if self.conf.weight_head:
                w_head = torch.abs(self.hyper_w_head(states))  # w_head:(bs, head_num)
                w_head = w_head.view(-1, self.n_head, 1).repeat(1, 1, self.n_agents)  # w_head: (bs,head_num,self.n_agents)
                head_attend *= w_head

            head_attend = torch.sum(head_attend, dim=1)

            if not self.conf.state_bias:
                v *= 0

            # regularize magnitude of attention logits 规范注意力机制对数的大小
            attend_mag_regs = self.attend_reg_coef * sum((logit ** 2).mean() for logit in head_attend_logits)
            head_entropies = [(-((probs + 1e-8).log() * probs).squeeze().sum(1).mean()) for probs in head_attend_weights]

            return head_attend, v, attend_mag_regs, head_entropies





