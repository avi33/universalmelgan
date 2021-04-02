import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.utils import weight_norm
import numpy as np
from mel2wav.layers import WNConv1d, WNConvTranspose1d, weights_init, ResnetBlock


class Generator(nn.Module):
    def __init__(self, input_size, ngf, n_residual_layers, ratios=[8, 6, 4]):
        super(Generator, self).__init__()        
        self.hop_length = np.prod(ratios)
        mult = int(2 ** len(ratios))

        model = [
            nn.ReflectionPad1d(3),            
            WNConv1d(input_size, mult * ngf, kernel_size=7, padding=0),
        ]

        # Upsample to raw audio scale
        for i, r in enumerate(ratios):
            model += [
                nn.LeakyReLU(0.2),
                WNConvTranspose1d(
                    mult * ngf,
                    mult * ngf // 2,
                    kernel_size=r * 2,
                    stride=r,
                    padding=r // 2 + r % 2,
                    output_padding=r % 2,
                ),
            ]

            for j in range(n_residual_layers):
                model += [ResnetBlock(mult * ngf // 2, dilation=3 ** j)]

            mult //= 2

        model += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(ngf, 1, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)
        self.apply(weights_init)

    def forward(self, x):
        y = self.model(x)
        return y

if __name__ == "__main__":
    b = 2 
    G = Generator(80, 64, 4)
    x = torch.randn(b, 80, 32)
    y = G(x)
    print(y.shape)    
    