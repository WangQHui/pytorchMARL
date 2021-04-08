import torch
import os
from pytorchMARL.network.base_NN import DRQN
from pytorchMARL.network.QMIX_NN import QmixNN

class QMIX:
    def __init__(self, args):
        self.args = args
        self.device = self.args.device
        self.n_actions = self.args.n_actions
        self.n_agents = self.args.n_agents
        self.state_shape = self.args.state_shape
        self.obs_shape = self.args.obs_shape
        input_shape = self.obs_shape

        # 根据参数决定DQRN的输入维度
        if self.args.last_action:
            input_shape += self.n_actions
        if self.args.reuse_network:
            input_shape += self.n_agents

        # 神经网络
        self.eval_drqn = DRQN(input_shape, self.args).to(self.device)
        self.target_drqn = DRQN(input_shape, self.args).to(self.device)

        self.eval_qmix = QmixNN(self.args).to(self.device)
        self.target_qmix = QmixNN(self.args).to(self.device)


        self.model_dir = self.args.model_dir

        if self.args.load_model:
            if os.path.exists(self.model_dir + '/1_drqn_net_params.pkl'):
                drqn_path = self.model_dir + '/1_drqn_net_params.pkl'
                qmix_path = self.model_dir + '/1_qmix_net_params.pkl'
                map_location = 'cuda:2' if self.args.cuda else 'cpu'
                self.eval_drqn.load_state_dict(torch.load(drqn_path, map_location=map_location))
                self.eval_qmix.load_state_dict(torch.load(qmix_path, map_location=map_location))
                print("successfully load models")
            else:
                raise Exception("No model!")

        # 使target网络和eval网络参数相同
        self.target_drqn.load_state_dict(self.eval_drqn.state_dict())
        self.target_qmix.load_state_dict(self.eval_qmix.state_dict())
        # 获取所有参数
        self.eval_parameters = list(self.eval_qmix.parameters()) + list(self.eval_drqn.parameters())
        # 获取优化器
        if self.args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)

        # 执行过程中，为每个agent维护一个eval_hidden
        # 学习时，为每个agent维护一个eval_hidden, target_hidden
        self.eval_hidden = None
        self.target_hidden = None
        print("Init qmix networks finished")

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
        :param batch: train data，obs: 四维（第几个episode，episode中的第几个transition，第几个agent，具体obs的维度）
        :param max_episode_len: max episode length
        :param train_step: step record for updating target network parameters
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

        s, s_, u, r, avail_u, avail_u_, terminated = batch['s'], batch['s_'], batch['u'], batch['r'], \
                                                     batch['avail_u'], batch['avail_u_'], batch['terminated']
        # 把填充经验的TD-error置0，防止影响学习
        mask = 1 - batch['padded'].float()

        # 得到每个agent当前与下个状态的Q值，维度为(episode个数, max_episode_len， n_agents， n_actions)
        q_evals, q_targets = self.get_q_values(batch, max_episode_len)
        s = s.to(self.device)
        u = u.to(self.device)
        r = r.to(self.device)
        s_ = s_.to(self.device)
        terminated = terminated.to(self.device)
        mask = mask.to(self.device)

        # 取每个agent动作对应的Q值，并且把最后不需要的一维去掉，因为最后一维只有一个值了
        # print("q_evals1 shape: ", q_evals.size()) #[batch_size, max_episode_len, n_agents, n_actions]
        q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)

        # 得到target_q，取所有行为中最大的 Q 值
        q_targets[avail_u_ == 0.0] = - 9999999
        q_targets = q_targets.max(dim=3)[0]
        # print("q_evals2 shape: ", q_evals.size()) # [batch_size, max_episode_len, n_agents]

        # qmix更新过程，evaluate网络输入的是每个agent选出来的行为的q值，target网络输入的是每个agent最大的q值，和DQN更新方式一样
        q_total_eval = self.eval_qmix(q_evals, s)
        q_total_target = self.target_qmix(q_targets, s_)

        # 计算一步 qmix的target
        targets = r + self.args.gamma * q_total_target * (1 - terminated)
        # 参数更新
        td_error = (q_total_eval - targets.detach())
        masked_td_error = mask * td_error

        # L2的损失函数，不能直接用mean，因为还有许多经验是没用的，所以要求和再比真实的经验数，才是真正的均值
        loss = (masked_td_error ** 2).sum() / mask.sum()
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        # 在指定周期更新 target network 的参数
        if train_step > 0 and train_step % self.args.update_target_params == 0:
            self.target_drqn.load_state_dict(self.eval_drqn.state_dict())
            self.target_qmix.load_state_dict(self.eval_qmix.state_dict())

    def init_hidden(self, episode_num):
        """
        为每个episode初始化一个eval_hidden,target_hidden
        :param episode_num:
        :return:
        """
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.drqn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.drqn_hidden_dim))

    def get_q_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        for transition in range(max_episode_len):
            # 为每个obs加上agent编号和last_action
            inputs, inputs_ = self._get_inputs(batch, transition)
            inputs = inputs.to(self.device)  # [batch_size*n_agents, obs_shape+n_agents+n_actions]
            inputs_ = inputs_.to(self.device)

            self.eval_hidden = self.eval_hidden.to(self.device)
            self.target_hidden = self.target_hidden.to(self.device)

            q_eval, self.eval_hidden = self.eval_drqn(inputs, self.eval_hidden)
            q_target, self.target_hidden = self.target_drqn(inputs, self.target_hidden)

            # 形状变化，把q_eval维度重新变回(8, 5,n_actions)
            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            # 添加transition信息
            q_evals.append(q_eval)
            q_targets.append(q_target)

        # 将 q_eval 和 q_target 列表中的max_episode_len个数组（episode_num, n_agents, n_actions)
        # 堆叠为(batch_size, max_episode_len, n_agents, n_actions)的数组
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        return q_evals, q_targets

    def get_inputs(self, batch, transition_idx):
        # 取出所有episode上该transition_idx的经验，onehot_u要用到上一条故取出所有
        obs, obs_, onehot_u = batch['o'][:, transition_idx],\
                              batch['o_'][:, transition_idx], batch['onehot_u'][:]
        inputs, inputs_ = [], []
        inputs.append(obs)
        inputs_.append(obs_)
        # 经验池的大小
        episode_num = obs.shape[0]

        # obs添上一个动作，agent编号
        if self.args.last_action:
            # 如果是第一条经验，就让前一个动作为0向量
            if transition_idx == 0:
                inputs.append(torch.zeros_like(onehot_u[:, transition_idx]))
            else:
                inputs.append(onehot_u[:, transition_idx-1])
            inputs_.append(onehot_u[:, transition_idx])
        if self.args.reuse_network:
            """
            因为当前的obs三维的数据，每一维分别代表(episode编号，agent编号，obs维度)，直接在dim_1上添加对应的向量即可，
            比如给agent_0后面加(1, 0, 0, 0, 0)，表示5个agent中的0号。
            而agent_0的数据正好在第0行，那么需要加的agent编号恰好就是一个单位矩阵，即对角线为1，其余为0
            """
            inputs.append(torch.eye(self.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            inputs_.append(torch.eye(self.n_agents).unsqueeze(0).expand(episode_num, -1, -1))

        # 把batch_size, n_agents的agents的obs拼接起来
        # 因为这里所有的所有agent共享一个神经网络，每条数据中带上了自己的编号，所以还是自己的数据
        # (batch_size, n_agents, n_actions) ->形状为(batch_size*n_agents, n_actions)
        inputs = torch.cat([x.reshape(episode_num*self.n_agents, -1) for x in inputs], dim=1)
        inputs_ = torch.cat([x.reshape(episode_num*self.n_agents, -1) for x in inputs_], dim=1)

        return inputs, inputs_

    def save_model(self, train_step):
        num = str(train_step // self.args.save_frequency)

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        print("save model: {} epoch.".format(num))

        torch.save(self.eval_drqn.state_dict(), self.model_dir+'/'+num+'_drqn_params.pkl')
        torch.save(self.eval_qmix.state_dict(), self.model_dir+'/'+num+'_qmix_params.pkl')






































