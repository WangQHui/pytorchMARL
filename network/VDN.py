import torch
import  torch.nn as nn

class VDN_NN(nn.Module):
    def __init__(self):
        super(VDN_NN, self).__init__()

    def forward(self, q_values):
        return torch.sum(q_values, dim=2, keepdim=True)