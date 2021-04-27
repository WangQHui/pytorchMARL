import torch
import os
from pytorchMARL.network.base_NN import DRQN
from pytorchMARL.network.ComaCritic import ComaCritic
from pytorchMARL.common.utils import td_lambda_target

class COMA:
    def __init__(self, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        actor_input_shape = self.obs_shape  # actor网络输入的维度，和vdn、qmix的rnn输入维度一样，使用同一个网络结构
        critic_input_shape = self._get_critic_input_shape()  # critic网络输入的维度

        # 根据参数决定RNN的输入维度
        if args.last_action:
            actor_input_shape += self.n_actions
        if args.reuse_network:
            actor_input_shape += self.n_agents

        # 神经网络
        # 每个agent选动作的网络,输出当前agent所有动作对应的概率，用该概率选动作的时候还需要用softmax再运算一次。
        if self.args.alg == 'coma':
            print('Init alg coma')
            self.eval_drqn = DRQN(actor_input_shape, args)
        else:
            raise Exception("No such algorithm")

        # 得到当前agent的所有可执行动作对应的联合Q值，得到之后需要用该Q值和actor网络输出的概率计算advantage
        self.eval_critic = ComaCritic(critic_input_shape, self.args)
        self.target_critic = ComaCritic(critic_input_shape, self.args)

        if self.args.cuda:
            self.eval_drqn.cuda()
            self.target_critic.cuda()
            self.eval_critic.cuda()

        self.model_dir = args.model_dir + '/' + args.alg + '/' + args.map
        # 如果存在模型则加载模型
        if self.args.load_model:
            if os.path.exists(self.model_dir + '/drqn_params.pkl'):
                path_drqn = self.model_dir + '/drqn_params.pkl'
                path_coma = self.model_dir + '/critic_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.eval_drqn.load_state_dict(torch.load(path_drqn, map_location=map_location))
                self.eval_critic.load_state_dict(torch.load(path_coma, map_location=map_location))
                print('Successfully load the model: {} and {}'.format(path_drqn, path_coma))
            else:
                raise Exception("No model!")

        # 让target_net和eval_net的网络参数相同
        self.target_critic.load_state_dict(self.eval_critic.state_dict())

        self.drqn_parameters = list(self.eval_drqn.parameters())
        self.critic_parameters = list(self.eval_critic.parameters())

        if args.optimizer == "RMS":
            self.critic_optimizer = torch.optim.RMSprop(self.critic_parameters, lr=args.lr_critic)
            self.drqn_optimizer = torch.optim.RMSprop(self.drqn_parameters, lr=args.lr_actor)

        # 执行过程中，要为每个agent都维护一个eval_hidden
        # 学习过程中，要为每个episode的每个agent都维护一个eval_hidden
        self.eval_hidden = None

    def _get_critic_input_shape(self):
        # state
        input_shape = self.state_shape
        # obs
        input_shape += self.obs_shape
        # agent_id
        input_shape += self.n_agents
        # 所有agent的当前动作和上一个动作
        input_shape += self.n_actions * self.n_agents * 2

        return input_shape

    def learn(self, batch, max_episode_len, train_step, epsilon):
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        u, r, avail_u, terminated = batch['u'], batch['r'], batch['avail_u'], batch['terminated']
        mask = (1 - batch["padded"].float()).repeat(1, 1, self.n_agents)  # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习
        if self.args.cuda:
            u = u.cuda()
            mask = mask.cuda()
        # 根据经验计算每个agent的Ｑ值,从而跟新Critic网络。然后计算各个动作执行的概率，从而计算advantage去更新Actor
        q_values = self._train_critic(batch, max_episode_len, train_step)  # 训练critic网络，并且得到每个agent的所有动作的Ｑ值
        action_prob = self._get_action_prob(batch, max_episode_len, epsilon)   # 每个agent的所有动作的概率

        q_taken = torch.gather(q_values, dim=3, index=u).squeeze(3)  # 每个agent的选择的动作对应的Ｑ值
        pi_taken = torch.gather(action_prob, dim=3, index=u).squeeze(3)  # 每个agent的选择的动作对应的概率
        pi_taken[mask == 0] = 1.0  # 因为要取对数，对于那些填充的经验，所有概率都为0，取了log就是负无穷了，所以让它们变成1
        log_pi_taken = torch.log(pi_taken)

        # 计算advantage
        baseline = (q_values * action_prob).sum(dim=3, keepdim=True).squeeze(3).detach()
        advantage = (q_taken - baseline).detach()
        loss = -((advantage * log_pi_taken) * mask).sum() / mask.sum()
        self.drqn_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.drqn_parameters, self.args.grad_norm_clip)
        self.drqn_optimizer.step()

    def _get_critic_inputs(self, batch, transition_idx, max_episode_len):
        # 取出所有episode上该transition_idx的经验
        obs, obs_, s, s_ = batch['o'][:, transition_idx], batch['o_'][:, transition_idx],\
                           batch['s'][:, transition_idx], batch['s_'][:, transition_idx]
        onehot_u = batch['onehot_u'][:, transition_idx]
        if transition_idx != max_episode_len - 1:
            onehot_u_ = batch['onehot_u_'][:, transition_idx + 1]
        else:
            onehot_u_ = torch.zeros(*onehot_u.shape)

        # s和s_next是二维的，没有n_agents维度，因为所有agent的s一样。其他都是三维的，到时候不能拼接，所以要把s转化成三维的



