import torch
import torch.nn as nn
import os
from pytorchMARL.network.base_NN import DRQN
from pytorchMARL.network.Qtran import Qtran_base, QtranV

class QtranBase:
    def __init__(self, conf):
        self.conf = conf
        self.n_agents = conf.n_agents
        self.n_actions = conf.n_actions
        self.state_shape = conf.state_shape
        self.obs_shape = conf.obs_shape
        input_shape = self.obs_shape

        # 根据参数决定DRQN的输入维度
        if self.conf.last_action:
            # 当前agent的上一个动作的独热码向量
            input_shape += self.n_actions
        if self.conf.reuse_network:
            input_shape += self.n_agents

        # 神经网络
        # 每个agent选动作的网络
        self.eval_drqn = DRQN(input_shape, conf).to(self.device)
        self.target_drqn = DRQN(input_shape, conf).to(self.device)
        # joint动作价值网络
        self.eval_joint_qtran = QtranBase(conf).to(self.device)
        self.target_joint_qtran = QtranBase(conf).to(self.device)

        self.v = QtranV(conf)

        self.model_dir = self.conf.model_dir + '/' + conf.alg + '/' + conf.map
        # 如果存在模型则加载模型
        if self.conf.load_model:
            if os.path.exists(self.model_dir + '/1_drqn_net_params.pkl'):
                drqn_path = self.model_dir + '/1_drqn_net_params.pkl'
                joint_qtran_path = self.model_dir + '/1_joint_qtran_net_params.pkl'
                v_path = self.model_dir + '/1_v_params.pkl'
                map_location = 'cuda:2' if self.conf.cuda else 'cpu'
                self.eval_drqn.load_state_dict(torch.load(drqn_path, map_location=map_location))
                self.eval_joint_qtran.load_state_dict(torch.load(joint_qtran_path, map_location=map_location))
                self.eval_v.load_state_dict(torch.load(v_path, map_location=map_location))
                print("successfully load models")
            else:
                raise Exception("No model!")

        # 让target_net和evel_net的网络参数相同
        self.target_drqn.load_state_dict(self.eval_drqn.state_dict())
        self.target_joint_qtran.load_state_dict(self.eval_joint_qtran.state_dict())

        # 获取所有参数
        self.eval_parameters = list(self.eval_drqn.parameters()) + \
                               list(self.eval_joint_qtran.parameters()) + \
                               list(self.v.parameters())
        # 获取优化器
        if conf.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=conf.lr)

        # 执行过程中，为每个agent维护一个eval_hidden
        # 学习时，为每个agent维护一个eval_hidden, target_hidden
        self.eval_hidden = None
        self.target_hidden = None
        print("Init qtran networks finished")


    # train_step表示是第几次学习，用来控制更新target_net网络的参数
    def learn(self, batch, max_episode_len, train_step, episode=None):
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
        episode_num = batch['o'].shape[0]
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
        individual_q_clone[avail_u] = - 999999
        individual_q_targets[avail_u_ == 0.0] = - 999999

        opt_onehot_eval = torch.zeros(*individual_q_clone.shape)
        opt_action_eval = individual_q_clone.argmax(dim=3, keepdim=True)
        opt_onehot_eval =opt_onehot_eval.scatter(-1, opt_action_eval[:, :].cpu(), 1)

        opt_onehot_target = torch.zeros(*individual_q_targets.shape)
        opt_action_target = individual_q_targets.argmax(dim=3, keepdim=True)
        opt_onehot_target = opt_onehot_target.scatter(-1, opt_action_target[:, :].cpu(), 1)

        # ------------------------ L_td 真实动作值的损失-------------------------------
        # 计算joint_q和v
        # joint_q、v的维度为(episode个数, max_episode_len, 1), 而且joint_q在后面的L_nopt还要用到
        joint_q_evals, joint_q_targets, v = self.get_qtran(batch, hidden_evals, hidden_targets, opt_onehot_target)

        # loss
        y_dqn = r.squeeze(-1) + self.conf.gamma * joint_q_targets * (1 - terminated.squeeze(-1))
        td_error = joint_q_evals - y_dqn.detach()
        l_td = ((td_error*mask) ** 2).sum() / mask.sum()
        # ------------------------ L_td -------------------------------

        # ------------------------ L_opt -------------------------------
        # 将局部最优动作的Q值相加
        # 这里要使用individual_q_clone，它把不能执行的动作Q值改变了，使用individual_q_evals可能会使用不能执行的动作的Q值
        # (episode个数, max_episode_len)
        q_sum_opt = individual_q_clone.max(dim=-1)[0].sum(dim=-1)

        # 重新得到joint_q_hat_opt，它和joint_q_evals的区别是前者输入的动作是局部最优动作，后者输入的动作是执行的动作
        # (episode个数, max_episode_len)
        joint_q_hat_opt, _, _ = self.get_qtran(batch, hidden_evals, hidden_targets, opt_onehot_eval, hat=True)
        opt_error = q_sum_opt - joint_q_hat_opt.detach() + v
        l_opt = ((opt_error*mask) ** 2).sum() / mask.sum()
        # ------------------------ L_opt -------------------------------

        # ------------------------ L_nopt -------------------------------
        # 每个agent的执行动作的Q值,(episode个数, max_episode_len, n_agents, 1)
        # (episode个数, max_episode_len)
        q_individual = torch.gather(individual_q_evals, dim=-1, index=u).squeeze(-1)
        q_sum_nopt = q_individual.sum(dim=-1)

        # 计算l_nopt时需要将joint_q_evals固定
        nopt_error = q_sum_nopt - joint_q_evals.detach() + v
        nopt_error = nopt_error.clamp(max=0)
        l_nopt = ((nopt_error*mask) ** 2).sum() / mask.sum()
        # ------------------------ L_nopt -------------------------------

        print('l_td is {}, l_opt is {}, l_nopt is {}'.format(l_td, l_opt, l_nopt))
        loss =l_td + self.conf.lambda_opt*l_opt + self.conf.lambda_nopt*l_nopt
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.eval_parameters, self.conf.grad_norm_clip)
        self.optimizer.step()

        if train_step > 0 and train_step % self.conf.update_target_params == 0:
            self.target_drqn.load_state_dict(self.eval_drqn.state_dict())
            self.target_joint_qtran.load_state_dict(self.eval_joint_qtran.state_dict())

    def init_hidden(self, episode_num):
        """
        为每个episode初始化一个eval_hidden,target_hidden
        :param episode_num:
        :return:
        """
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.conf.dqrn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.conf.dqrn_hidden_dim))

    def _get_individual_q(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        q_evals, q_targets, hidden_evals, hidden_target = [], [], [], []
        for transition in range(max_episode_len):
            # 给obs加last_action、agent_id
            inputs, inputs_ = self.get_inputs(batch, transition)
            inputs = inputs.to(self.device)  # [batch_size*n_agents, obs_shape+n_agents+n_actions]
            inputs_ = inputs_.to(self.device)

            self.eval_hidden = self.eval_hidden.to(self.device)
            self.target_hidden = self.target_hidden.to(self.device)

            # 要用第一条经验把target网络的hidden_state初始化好，直接用第二条经验传入target网络不对
            if transition == 0:
                _, self.target_hidden = self.target_drqn(inputs, self.eval_hidden)
            # inputs维度为(40,96)，得到的q_eval维度为(40,n_actions)
            q_evals, self.eval_hidden = self.eval_drqn(inputs, self.eval_hidden)
            q_targets, self.target_hidden = self.target_drqn(inputs_, self.target_hidden)
            hidden_evals, hidden_target = self.eval_hidden.clone(), self.target_hidden.clone()

            # 把q_eval维度重新变回(8, 5,n_actions)
            q_evals = q_evals.view(episode_num, self.n_agents, -1)
            q_targets = q_targets.view(episode_num, self.n_agents, -1)
            hidden_evals = hidden_evals.view(episode_num, self.n_agents, -1)
            hidden_target = not hidden_target.view(episode_num, self.n_agents, -1)
            # 添加transition信息
            q_evals.append(q_evals)
            q_targets.append(q_targets)
            hidden_evals.append(hidden_evals)
            hidden_target.append(hidden_target)

        # 将 q_eval 和 q_target 列表中的max_episode_len个数组（episode_num, n_agents, n_actions)
        # 堆叠为(batch_size, max_episode_len, n_agents, n_actions)的数组
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        hidden_evals = torch.stack(hidden_evals, dim=1)
        hidden_target = torch.stack(hidden_target, dim=1)
        return q_evals, q_targets, hidden_evals, hidden_target

    def get_inputs(self, batch, transition_idx):
        # 取出所有episode上该transition_idx的经验，onehot_u要用到上一条故取出所有
        obs, obs_, onehot_u = batch['o'][:, transition_idx], \
                              batch['o_'][:, transition_idx], batch['onehot_u'][:]
        episode_num = batch['o'].shape[0]
        inputs, inputs_ = [], []
        inputs.append(obs)
        inputs_.append(obs_)

        # 为每个obs加上agent编号和last_action
        if self.conf.last_action:
            # 如果是第一条经验，就让前一个动作为0向量
            if transition_idx == 0:
                inputs.append(torch.zeros_like(onehot_u[:, transition_idx]))
            else:
                inputs.append(onehot_u[:, transition_idx - 1])
            inputs_.append(onehot_u[:, transition_idx])
        if self.conf.reuse_network:
            """
            因为当前的obs三维的数据，每一维分别代表(episode编号，agent编号，obs维度)，直接在dim_1上添加对应的向量即可，
            比如给agent_0后面加(1, 0, 0, 0, 0)，表示5个agent中的0号。
            而agent_0的数据正好在第0行，那么需要加的agent编号恰好就是一个单位矩阵，即对角线为1，其余为0
            """
            inputs.append(torch.eye(self.conf.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            inputs_.append(torch.eye(self.conf.n_agents).unsqueeze(0).expand(episode_num, -1, -1))

        # 把batch_size, n_agents的agents的obs拼接起来
        # 因为这里所有的所有agent共享一个神经网络，每条数据中带上了自己的编号，所以还是自己的数据
        # (batch_size, n_agents, n_actions) ->形状为(batch_size*n_agents, n_actions)
        inputs = torch.cat([x.reshape(episode_num * self.n_agents, -1) for x in inputs], dim=1)
        inputs_ = torch.cat([x.reshape(episode_num * self.n_agents, -1) for x in inputs_], dim=1)

    def get_qtran(self, batch, hidden_evals, hidden_targets, local_opt_actions, hat=False):
        episode_num, max_episode_len, _, _ = hidden_targets.shape
        s = batch['s'][:, :max_episode_len]
        s_ = batch['s_'][:, :max_episode_len]
        onehot_u = batch['onehot_u'][:, max_episode_len]

        s = s.to(self.device)
        s_ = s_.to(self.device)
        onehot_u = onehot_u.to(self.device)
        hidden_evals = hidden_evals.to(self.device)
        hidden_targets = hidden_targets.to(self.device)
        local_opt_actions = local_opt_actions.to(self.device)

        if hat:
            # 神经网络输出的q_eval、q_target、v的维度为(episode_num * max_episode_len, 1)
            q_evals = self.eval_joint_qtran(s, hidden_evals, local_opt_actions)
            q_targets = None
            v = None

            # 把q_eval维度变回(episode_num, max_episode_len)
            q_evals = q_evals.view(episode_num, -1, 1).squeeze(-1)

        else:
            q_evals = self.eval_joint_qtran(s, hidden_evals, onehot_u)
            q_targets = self.eval_joint_qtran(s_, hidden_targets, local_opt_actions)
            v = self.V(s, hidden_evals)
            # 把q_eval、q_target、v维度变回(episode_num, max_episode_len)
            q_evals = q_evals.view(episode_num, -1, 1).squeeze(-1)
            q_targets = q_targets.view(episode_num, -1, 1).squeeze(-1)
            v = v.view(episode_num, -1, 1).squeeze(-1)

        return q_evals, q_targets, v

    def save_model(self, train_step):
        num = str(train_step // self.conf.save_frequency)

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        print("save model: {} epoch.".format(num))

        torch.save(self.eval_drqn.state_dict(), self.model_dir + '/' + num + '_drqn_params.pkl')
        torch.save(self.eval_joint_qtran.state_dict(), self.model_dir + '/' + num + '_joint_qtran_params.pkl')
        torch.save(self.v.state_dict(), self.model_dir + '/' + num + '_v_params.pkl')































