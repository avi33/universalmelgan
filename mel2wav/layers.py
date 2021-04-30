import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
  
def weights_init(m):
    if isinstance(m, torch.nn.Conv1d):
        m.weight.data.normal_(0.0, 0.02)    
        m.bias.data.fill_(0)
    
    elif isinstance(m, torch.nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)    
        m.bias.data.fill_(0)

    elif isinstance(m, torch.nn.ConvTranspose1d):
        m.weight.data.normal_(0.0, 0.02)    
        m.bias.data.fill_(0)

    elif isinstance(m, torch.nn.ConvTranspose2d):
        m.weight.data.normal_(0.0, 0.02)    
        m.bias.data.fill_(0)

def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))

def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))

def WNConv2d(*args, **kwargs):
    return weight_norm(nn.Conv2d(*args, **kwargs))

class GatedTanh(nn.Module):
    def __init__(self):
        super(GatedTanh, self).__init__()
    def forward(self, x):
        s, t = x.split(x.size(1) // 2, dim=1)
        y = torch.sigmoid(s) * torch.tanh(t)
        return y

class GatedTanhWNConv(nn.Module):
    def __init__(self, 
    in_channels, 
    out_channels, 
    kernel_size, 
    stride=1,
    dilation=1,
    padding=0,
    groups=1):
        super(GatedTanhWNConv, self).__init__()
        self.conv = WNConv1d(in_channels, out_channels*2, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups)        
        self.act = GatedTanh()        

    def forward(self, x):
        c = self.conv(x)        
        y = self.act(c)
        return y
    
class FastGlobalAvgPool2d(nn.Module):
    def __init__(self, flatten=False):
        super(FastGlobalAvgPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(dilation),
            WNConv1d(dim, dim, kernel_size=3, dilation=dilation),            
            nn.LeakyReLU(0.2),
            GatedTanhWNConv(dim, dim, kernel_size=1)
        )
        self.shortcut = WNConv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)