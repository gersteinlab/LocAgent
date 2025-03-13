import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, v_ft, u_ft):
        super().__init__()
        self.bilinear = nn.Bilinear(v_ft, u_ft, 1)
        self.act = nn.Sigmoid()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
       
    def forward(self, v_h, c):
        
#         c = self.act(c)
#         v_h = self.act(v_h)
        
#         c = c.expand_as(v_h)
        sc_1 = self.bilinear(v_h, c)

        return sc_1.squeeze()


