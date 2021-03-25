import numpy as np
import torch
from pytorchMARL.policy.qmix import QMIX

class Agents:
    def __init__(self, conf):
        self.conf = conf
        self.device = conf.device
        self.n_agents = conf.n_agents
        self.n_actions = conf.n_actions
        self.state_shape = conf.state_shape
        self.obs_shape = conf.obs_shape
        self.episode_limit = conf.episode_limit

        self.policy = QMIX(conf)

        print("Agents inited!")

    def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, evaluate=False):
        #inputs = obs.copy()
        # 可供选择的动作
        avail_actions_idx = np.nonzero(avail_actions)[0]
        # agent索引为独热码
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1
        if self.conf.last_action:
            obs = np.hstack((obs, last_action))
        if self.conf.reuse_network:
            obs = np.hstack((obs, agent_id))
        hidden_state = self.policy.eval_hidden[:, agent_num, :]

        # 转置
        obs = torch.tensor(obs).unsqueeze(0).to(self.device)
        avail_actions = torch.tensor(avail_actions).unsqueeze(0).to(self.device)

        # 获取Q(s, a)
        q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_drqn(obs, hidden_state)
        # 不可选的动作 q 值设为无穷小
        q_value[avail_actions == 0.0] = -float("inf")
        if np.random.uniform() < epsilon:
            return np.random.choice(avail_actions_idx)
        else:
            return torch.argmax(q_value)

    def get_max_episode_len(self, batch):
        terminated = batch["terminated"]
        max_episode_len = 0
        for episode_idx in range(terminated.shape[0]):
            for transition_idx in range(self.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        return max_episode_len

    def train(self, batch, train_step, epsilon=None):
        # 不同的episode的数据长度不同，因此需要得到最大长度
        max_episode_len = self.get_max_episode_len(batch)
        for key in batch.keys():
            batch[key] = batch[key][:, :max_episode_len]
        self.policy.learn(batch, max_episode_len, train_step, epsilon)
        if train_step > 0 and train_step % self.conf.save_frequency == 0:
            self.policy.save_model(train_step)



