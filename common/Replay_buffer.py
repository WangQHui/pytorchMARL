import numpy as np
import threading

class ReplayBuffer:
    def __init__(self, conf):
        self.conf = conf
        self.n_agents = conf.n_agents
        self.n_actions = conf.n_actions
        self.state_shape = conf.state_shape
        self.obs_shape = conf.obs_shape
        self.episode_limit = conf.episode_limit
        self.size = conf.buffer_size

        self.buffer = {
            'o': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape]),
            'u': np.empty([self.size, self.episode_limit, self.n_agents, 1]),
            's': np.empty([self.size, self.episode_limit, self.state_shape]),
            'r': np.empty([self.size, self.episode_limit, 1]),
            'o_': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape]),
            'u_': np.empty([self.size, self.episode_limit, self.n_agents, 1]),
            'avail_u': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
            'avail_u_': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
            'terminated': np.empty([self.size, self.episode_limit, 1]),
            'padded': np.empty([self.size, self.episode_limit, 1])
        }
        self.lock = threading.lock()
        print("Replay Buffer inited!")

    def store(self, episode_batch):
        with self.lock:
            # 这批数据有多少
            batch_size = episode_batch['o'].shape[0]
            # 获取可以写入的位置
            idxes = self.get_storage_idx(inc=batch_size)
            # 存储经验
            for key in self.buffer.keys():
                self.buffer[key][idxes] = episode_batch[key]
            # 更新当前buffer的大小

    # 采样方法 .sample() 用来在训练过程中随机的选择一个迁移批次(batch of transitions)
    def sample(self, batch_size):
        """
        采样部分episode
        :param batch_size:
        :return:
        """
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffers.keys():
            temp_buffer[key] = self.buffer[key][idx]
        return temp_buffer

    def get_storage_idx(self, inc=None):
        """
        得到可以填充的索引数组
        :param inc:
        :return: 索引数组
        """
        inc = inc or 1
        # 如果保存后不超过 buffer 的大小，则返回当前已填充到的索引+1开始 inc 长度的索引数组
        if self.current_idx + inc <= self.size:
            idx = np.arange(self.current_idx, self.current_idx+inc)
            self.current_idx += inc
        # 如果剩下位置不足以保存，但 current_idx 还没到末尾，则填充至末尾后，从头开始覆盖
        elif self.current_idx < self.size:
            overflow = inc - (self.size - self.current_idx)
            idx_a = np.arange(self.current_idx, self.size)
            idx_b = np.arange(0, overflow)
            self.current_idx = overflow
        # 否则直接从头开始覆盖
        else:
            idx = np.arange(0, inc)
            self.current_idx = inc

        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx





