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
        self.eval_joint_qtran = Qatten(args)
        self.target_joint_qtran = Qatten(args)

        if self.args.cuda():
            self.eval_drqn.cuda()
            self.target_drqn.cuda()
            self.eval_qatten_net.cuda()
            self.target_qatten_net.cuda()

        self.model_dir = args.model_dir + '/' + args.alg + '/' + args.map

        # 如果存在模型则加载模型
        if self.args.load_model:
            if os.path.exists(self.model_dir + '/drqn_net_params.pkl'):
                path_drqn = self.model_dir + '/drqn_net_params.pkl'
                path_qatten = self.model_dir + '/qatten_net_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.eval_drqn.load_state_dict(torch.load(path_drqn, map_location=map_location))
                self.eval_qatten_net.load_state_dict(torch.load(path_qatten, map_location=map_location))
                print('Successfully load the model: {} and {}'.format(path_drqn, path_qatten))
            else:
                raise Exception("No model!")

