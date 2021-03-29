import torch
import torch.nn as nn
import torch.functional as F

# 加入动作价值网络，输入（state, hidden_state, actions, q_value)


class Qtran_base(nn.Module):
    def __init__(self, conf):
        super(Qtran_base, self).__init__()
        self.conf = conf

        # action_encoding对输入的每个agent的hidden_state进行编码，
        # 将所有agent的hidden_state和动作相加得到近似的联合hidden_state和动作
        ae_input = conf.rnn_hidden_dim + conf.n_actions
        self.hidden_action_encoding = nn.Sequential(nn.Linear(ae_input, ae_input),

                                                    nn.ReLU(),
                                                    nn.Linear(ae_input, ae_input))

        # 编码求和之后输入state、所有agent的hidden_state和动作之和
        q_input_size = conf.state_shape + conf.n_actions + conf.rnn_hidden_dim
        self.Q = nn.Sequential(nn.Linear(q_input_size, conf.qtran_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(conf.qtran_hidden_dim, conf.qtran_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(conf.qtran_hidden_dim, 1))

    # 因为所有时刻所有agent的hidden_state在之前已经计算好，
    # 所以联合Q值可以一次性计算所有transition的
    def forward(self, state, hidden_states, actions):
        episode_num, max_episode_len, n_agents, _ = actions.shape
        hidden_actions = torch.cat([hidden_states, actions], dim=-1)
        hidden_actions = hidden_actions.reshape(-1, self.conf.rnn_hidden_dim + self.conf.n_actions)
        hidden_actions_encoding = self.hidden_action_encoding(hidden_actions)
        # 变回n_agents维度求和
        hidden_actions_encoding = hidden_actions_encoding.reshape(episode_num*max_episode_len, n_agents, -1).sum(dim=-2)

        inputs = torch.cat([state.reshape(episode_num*max_episode_len, -1), hidden_actions_encoding], dim=-1)
        q = self.Q(inputs)
        return q


class QtranV(nn.Module):
    def __init__(self, conf):
        super(QtranV, self).__init__()
        self.conf = conf

        # hidden_encoding对输入的的每个agent的hidden_state进行编码，
        # 将所有agent的hidden_state相加得到近似的联合hidden_state
        hidden_input = conf.rnn_hidden_dim
        self.hidden_encoding = nn.Sequential(nn.Linear(hidden_input, hidden_input),
                                             nn.ReLU(),
                                             nn.Linear(hidden_input, hidden_input))

        # 编码求和之后输入state、所有agent的hidden_state之和
        v_input_size = conf.state_shape + conf.rnn_hidden_dim
        self.V = nn.Sequential(nn.Linear(v_input_size, conf.qtran_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(conf.qtran_hidden_dim, conf.qtran_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(conf.qtran_hidden_dim, 1))

    def forward(self, state, hidden):
        episode_num, max_episode_len, n_agents, _ = hidden.shape
        state = state.reshape(episode_num*max_episode_len, -1)
        hidden_encoding = self.hidden_encoding(hidden.reshape(-1, self.conf.rnn_hidden_dim))
        hidden_encoding = hidden_encoding.reshape(episode_num*max_episode_len, n_agents, -1).sum(dim=-2)

        inputs = torch.cat([state, hidden_encoding], dim=-1)
        v = self.V(inputs)

        return v
