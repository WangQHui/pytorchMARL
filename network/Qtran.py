import torch
import torch.nn as nn
import torch.functional as F

class Qtran_base(nn.Module):
    def __init__(self, conf):
        super(Qtran_base, self).__init__()
        self.conf = conf

        
