import torch.nn as nn
import torch
import torch.nn.functional as F

class QmixNN(nn.Module):
    def __init__(self, args):
        super(QmixNN, self).__init__()
        self.args = args
        """
                生成的hyper_w1需要是一个矩阵，但是torch NN的输出只能是向量；
                因此先生成一个（行*列）的向量，再reshape
                
                args.n_agents是使用hyper_w1作为参数的网络的输入维度， args.qmix_hidden_dim是网络隐藏层参数个数
                从而经过hyper_w1得到（经验条数， args.n_agents * args.qmix_hidden_dim)的矩阵
        """
        if args.two_hyper_layers:
            self.hyper_w1 = nn.Sequential(nn.Linear(args.state_shape, args.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(args.hyper_hidden_dim, args.n_agents * args.qmix_hidden_dim))
            # 经过hyper_w2 得到的（经验条数， 1）的矩阵
            self.hyper_w2 = nn.Sequential(nn.Linear(args.state_shape, args.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(args.hyper_hidden_dim, args.qmix_hidden_dim))

        else:
            self.hyper_w1 = nn.Linear(self.state_shape, args.n_agents * args.qmix_hidden_dim)
            # 经过hyper_w2 得到的（经验条数， 1）的矩阵
            self.hyper_w2 = nn.Linear(self.state_shape, args.qmix_hidden_dim)

        # hyper_b1 得到的（经验条数，args.qmix_hidden_dim)矩阵需要同样维度的hyper_b1
        self.hyper_b1 == nn.Linear(args.state_shape, args.qmix_hidden_dim)
        # hyper_b1 得到的（经验条数， 1）的矩阵需要同样维度的hyper_b1
        self.hyper_b2 == nn.Sequential(nn.Linear(args.state_shape, args.qmix_hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(args.qmix_hidden_dim, 1))


    """
        input:(batch_size, n_agents, qmix_hidden_dim)
        q_values:(episode_num, max_episode_Len, n_agents)
        states shape:(episode_num, max_episode_len, state_shape)
    """
    def forward(self, q_values, states):
        # print(args.state_shape)
        episode_num = q_values.size(0)
        q_values = q_values.view(-1, 1, self.conf.n_agents)  # (episode_num * max_episode_len, 1, n_agents)
        states = states.reshape(-1, self.conf.state_shape)  # (episode_num * max_episode_len, state_shape)

        w1 = torch.abs(self.hyper_w1(states))  # abs输出绝对值，保证w非负
        b1 = self.hyper_b1(states)  # 不需要进行限制

        w1 = w1.view(-1, self.n_agents, self.conf.qmix_hidden_dim)
        b1 = b1.view(-1, 1, self.conf.qmix_hidden_dim)

        hidden = F.elu(torch.bmm(q_values, w1) + b1)

        w2 = torch.abs(self.hyper_w2(states))
        b2 = self.hyper_b2(states)

        w2 = w2.view(-1, self.args.qmix_hidden_dim, 1)
        b2 = b2.view(-1, 1, 1)

        q_total = torch.bmm(hidden, w2) + b2
        q_total = q_total.view(episode_num, -1, 1)
        return q_total

