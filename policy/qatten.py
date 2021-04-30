import torch
import torch.nn as nn
import os
from pytorchMARL.network.base_NN import DRQN
from pytorchMARL.network.Qatten import Qatten

class Qatten:
    def __init__(self, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        input_shape = self.obs_shape

        # 根据参数决定DRQN的输入维度
        if self.args.last_action:
            # 当前agent的上一个动作的独热码向量
            input_shape += self.n_actions
        if self.args.reuse_network:
            input_shape += self.n_agents

        # 神经网络
        # 每个agent选动作的网络
        self.eval_drqn = DRQN(input_shape, args)
        self.target_drqn = DRQN(input_shape, args)
        # joint动作价值网络
        self.eval_joint_q = Qatten(args)
        self.target_joint_q = Qatten(args)

        if self.args.cuda():
            self.eval_drqn.cuda()
            self.target_drqn.cuda()
            self.eval_joint_q.cuda()
            self.target_joint_q.cuda()

        self.model_dir = args.model_dir + '/' + args.alg + '/' + args.map

        # 如果存在模型则加载模型
        if self.args.load_model:
            if os.path.exists(self.model_dir + '/drqn_net_params.pkl'):
                path_drqn = self.model_dir + '/drqn_net_params.pkl'
                path_qatten = self.model_dir + '/qatten_net_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.eval_drqn.load_state_dict(torch.load(path_drqn, map_location=map_location))
                self.eval_joint_q.load_state_dict(torch.load(path_qatten, map_location=map_location))
                print('Successfully load the model: {} and {}'.format(path_drqn, path_qatten))
            else:
                raise Exception("No model!")

        # 让target_net和evel_net的网络参数相同
        self.target_drqn.load_state_dict(self.eval_drqn.state_dict())
        self.target_joint_q.load_state_dict(self.eval_joint_q.state_dict())

        self.eval_parameters = list(self.eval_joint_q.parameters()) + \
                               list(self.eval_drqn.parameters())

        if args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)

        # 执行过程中，为每个agent维护一个eval_hidden
        # 学习时，为每个agent维护一个eval_hidden, target_hidden
        self.eval_hidden = None
        self.target_hidden = None
        print("Init qatten networks finished")

    # train_step表示是第几次学习，用来控制更新target_net网络的参数
    def learn(self, batch, max_episode_len, train_step, epsilon=None):
        """
        在learn的时候，抽取到的数据是四维的，四个维度分别为
        1——第几个episode
        2——episode中第几个transition
        3——第几个agent的数据
        4——具体obs维度。
        因为在选动作时不仅需要输入当前的inputs，还要给神经网络输入hidden_state，
        hidden_state和之前的经验相关，因此就不能随机抽取经验进行学习。所以这里一次抽取多个episode，
        然后一次给神经网络传入每个episode的同一个位置的transition
        :param batch:
        :param max_episode_len:
        :param train_step:
        :param episode:
        :return:
        """
        # 获得episode的数目
        episode_num = batch['o'].shape[0]
        # 初始化隐藏状态
        self.init_hidden(episode_num)
        # 把batch里的数据转化为tensor
        for key in batch.keys():
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        u, r, avail_u, avail_u_, terminated = batch['u'], batch['r'], batch['avail_u'], \
                                              batch['avail_u_'], batch['terminated']
        # 用来把那些填充的经验的TD-error置0，从而不影响到学习
        mask = (1 - batch["padded"].float()).squeeze(-1)
        # 得到每个agent当前的Q值和hidden_states，维度为(episode个数, max_episode_len， n_agents， n_actions/hidden_dim)
        q_evals, q_targets = self.get_q_values(batch, max_episode_len)
        u = u.to(self.device)
        r = r.to(self.device)
        avail_u = avail_u.to(self.device)
        avail_u_ = avail_u_.to(self.device)
        terminated = terminated.to(self.device)
        mask = mask.to(self.device)

        # 得到每个agent对应的Q和hidden_states，维度为(episode个数, max_episode_len, n_agents, n_actions/hidden_dim)
        individual_q_evals, individual_q_targets, hidden_evals, hidden_targets = self._get_individual_q(batch, max_episode_len).squeeze(3)

        # 得到当前时刻和下一时刻每个agent的局部最优动作及其one_hot表示
        individual_q_clone = individual_q_evals.clone()
        individual_q_clone[avail_u == 0.0] = - 999999
        individual_q_targets[avail_u_ == 0.0] = - 999999

        joint_q_evals, joint_q_targets = self.get_qatten(batch, hidden_evals, hidden_targets)

        # loss
        y_dqn = r.squeeze(-1) + self.args.gamma * joint_q_targets * (1 - terminated.squeeze(-1))
        td_error = joint_q_evals - y_dqn.detach()
        loss = ((td_error * mask) ** 2).sum() / mask.sum() + q_attend_regs



    def init_hidden(self, episode_num):
        """
        为每个episode初始化一个eval_hidden,target_hidden
        :param episode_num:
        :return:
        """
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.drqn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.drqn_hidden_dim))

    def get_inputs(self, batch, transition_idx):
        # 取出所有episode上该transition_idx的经验，onehot_u要用到上一条故取出所有
        obs, obs_, onehot_u = batch['o'][:, transition_idx], \
                              batch['o_'][:, transition_idx], batch['onehot_u'][:]
        episode_num = batch['o'].shape[0]
        inputs, inputs_ = [], []
        inputs.append(obs)
        inputs_.append(obs_)

        # 为每个obs加上agent编号和last_action
        if self.args.last_action:
            # 如果是第一条经验，就让前一个动作为0向量
            if transition_idx == 0:
                inputs.append(torch.zeros_like(onehot_u[:, transition_idx]))
            else:
                inputs.append(onehot_u[:, transition_idx - 1])
            inputs_.append(onehot_u[:, transition_idx])
        if self.args.reuse_network:
            """
            因为当前的obs三维的数据，每一维分别代表(episode编号，agent编号，obs维度)，直接在dim_1上添加对应的向量即可，
            比如给agent_0后面加(1, 0, 0, 0, 0)，表示5个agent中的0号。
            而agent_0的数据正好在第0行，那么需要加的agent编号恰好就是一个单位矩阵，即对角线为1，其余为0
            """
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            inputs_.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))

        # 把batch_size, n_agents的agents的obs拼接起来
        # 因为这里所有的所有agent共享一个神经网络，每条数据中带上了自己的编号，所以还是自己的数据
        # (batch_size, n_agents, n_actions) ->形状为(batch_size*n_agents, n_actions)
        inputs = torch.cat([x.reshape(episode_num * self.n_agents, -1) for x in inputs], dim=1)
        inputs_ = torch.cat([x.reshape(episode_num * self.n_agents, -1) for x in inputs_], dim=1)

    def save_model(self, train_step):
        num = str(train_step // self.args.save_frequency)

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        print("save model: {} epoch.".format(num))

        torch.save(self.eval_drqn.state_dict(), self.model_dir + '/' + num + '_drqn_params.pkl')
        torch.save(self.eval_joint_q.state_dict(), self.model_dir + '/' + num + '_joint_qatten_params.pkl')
