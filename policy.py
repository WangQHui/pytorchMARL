import torch
import os
from network.base_NN import DRQN
from network.QMIX_NN import QmixNN

class QMIX:
    def __init__(self, conf):
        self.conf = conf
        self.device = self.conf.device
        self.n_actions = self.conf.n_actions
        self.n_agents = self.conf.n_agents
        self.state_shape = self.conf.state_shape
        self.obs_shape = self.conf.obs_shape
        input_shape = self.obs_shape

        # 根据参数决定DQRN的输入维度
        if self.conf.last_action:
            input_shape += self.n_actions
        if self.conf.reuse_network:
            input_shape += self.n_agents

        # 神经网络
        self.eval_drqn = DRQN(input_shape, self.conf).to(self.device)
        self.target_drqn = DRQN(input_shape, self.conf).to(self.device)

        self.eval_qmix = QmixNN(self.conf).to(self.device)
        self.target_qmix = QmixNN(self.conf).to(self.device)


        self.model_dir = self.conf.model_dir

        if self.conf.load_model:
            if os.path.exists(self.model_dir + '/1_drqn_net_params.pkl'):
                drqn_path = self.model_dir + '/1_drqn_net_params.pkl'
                qmix_path = self.model_dir + '/1_qmix_net_params.pkl'
                map_location = 'cuda:2' if self.conf.cuda else 'cpu'
                self.eval_drqn.load_state_dict(torch.load(drqn_path, map_location=map_location))
                self.eval_qmix.load_state_dict(torch.load(qmix_path, map_location=map_location))
                print("successfully load models")
            else:
                raise Exception("No model!")

        # 使target网络和eval网络参数相同
        self.target_drqn.load_state_dict(self.eval_drqn.state_dict())
        self.target_qmix.load_state_dict(self.eval_qmix.state_dict())

        self.eval_parameters = list(self.eval_qmix.parameters()) + list(self.eval_drqn.parameters())

        if args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=self.conf.lr)

        # 执行过程中，为每个agent维护一个eval_hidden
        # 学习时，为每个agent维护一个eval_hidden, target_hidden
        self.eval_hidden = None
        self.target_hidden = None
        print("Init qmix networks finished")

    def learn(self, batch, max_episode_len, train_step, episode=None):
        """

        :param batch: train data，obs: 四维（第几个episode，episode中的第几个tarnsition，第几个agent，具体obs的维度）
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

        # 得到每个agent对应的Q值，维度为(episode个数, max_episode_len， n_agents， n_actions)
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
        q_targets[avail_u_ == 0.0] = -9999999
        q_targets = q_targets.max(dim=3)[0]
        # print("q_evals2 shape: ", q_evals.size()) # [batch_size, max_episode_len, n_agents]

        # qmix更新过程，evaluate网络输入的是每个agent选出来的行为的q值，target网络输入的是每个agent最大的q值，和DQN更新方式一样
        q_total_eval = self.eval_qmix(q_evals, s)
        q_total_target = self.target_qmix(q_targets, s_)

        targets = r + self.conf.gamma * q_total_target * (1 - terminated)
        # 参数更新
        td_error = (q_total_eval - targets.detach())
        masked_td_error = mask * td_error

        # 不能直接用mean，因为还有许多经验是没用的，所以要求和再比真实的经验数，才是真正的均值
        loss = (masked_td_error ** 2).sum() / mask.sum()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.conf.grad_norm_clip)
        self.optimizer.step()

        # 在指定周期更新 target network 的参数
        if train_step > 0 and train_step % self.conf.update_target_params == 0:
            self.target_drqn.load_state_dict(self.eval_drqn.state_dict())
            self.target_qmix.load_state_dict(self.eval_qmix.state_dict())

























