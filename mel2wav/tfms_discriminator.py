import torch
import torch.nn as nn
import torch.nn.functional as F
from mel2wav.layers import weights_init, WNConv1d, WNConv2d

class STFT(nn.Module):
    def __init__(self, fft_size, hop_size, win_length):
        super(STFT, self).__init__()
        self.fft_size = fft_size        
        self.hop_size = hop_size
        self.win_length = win_length
        window = "hann_window"
        self.register_buffer("window", getattr(torch, window)(win_length))

    def forward(self, x):
      p = (self.fft_size - self.hop_size) // 2
      x = F.pad(x, (p, p), "reflect").squeeze(1)
      x_stft = torch.stft(x, self.fft_size, self.hop_size, self.win_length, self.window, return_complex=False)    
      real, imag = x_stft.unbind(-1)
      mag = torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7))
      return mag

class NLayerDiscriminator(nn.Module):
    def __init__(self, ndf, n_layers, downsampling_factor):
        super(NLayerDiscriminator, self).__init__()
        model = nn.ModuleDict()

        model["layer_0"] = nn.Sequential(
            nn.ReflectionPad1d(7),
            WNConv1d(1, ndf, kernel_size=15),
            nn.LeakyReLU(0.2, True),
        )

        nf = ndf
        stride = downsampling_factor
        for n in range(1, n_layers + 1):
            nf_prev = nf
            nf = min(nf * stride, 1024)

            model["layer_%d" % n] = nn.Sequential(
                WNConv1d(
                    nf_prev,
                    nf,
                    kernel_size=stride * 10 + 1,
                    stride=stride,
                    padding=stride * 5,
                    groups=nf_prev // 4,
                ),
                nn.LeakyReLU(0.2, True),
            )

        nf = min(nf * 2, 1024)
        model["layer_%d" % (n_layers + 1)] = nn.Sequential(
            WNConv1d(nf_prev, nf, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
        )

        model["layer_%d" % (n_layers + 2)] = WNConv1d(
            nf, 1, kernel_size=3, stride=1, padding=1
        )

        self.model = model

    def forward(self, x):
        results = []
        for key, layer in self.model.items():
            x = layer(x)
            results.append(x)
        return results


class TimeDiscriminator(nn.Module):
    def __init__(self, num_D, ndf, n_layers, downsampling_factor):
        super(TimeDiscriminator, self).__init__()
        self.model = nn.ModuleDict()
        for i in range(num_D):
            self.model[f"disc_{i}"] = NLayerDiscriminator(
                ndf, n_layers, downsampling_factor
            )

        self.downsample = nn.AvgPool1d(4, stride=2, padding=1, count_include_pad=False)
        self.apply(weights_init)

    def forward(self, x):
        results = []
        for key, disc in self.model.items():
            results.append(disc(x))
            x = self.downsample(x)
        return results


class DBlock(nn.Module):
    def __init__(self):
        super(DBlock, self).__init__()
        model = [
            WNConv2d(1, 32, 3, 1),
            nn.LeakyReLU(0.2, True)            
        ]
        for _ in range(3):
            model += [
                WNConv2d(32, 32, kernel_size=3, stride=(1, 2)),    
                nn.LeakyReLU(0.2, True)
            ]
        model += [
            WNConv2d(32, 32, 1, 1),
            nn.LeakyReLU(0.2, True),
            WNConv2d(32, 1, 1, 1),
        ]
        self.block = nn.Sequential(*model)
        self.apply(weights_init)

    def forward(self, x):
        x = x.unsqueeze(1)
        y = self.block(x)
        return y


class FreqDiscriminator(nn.Module):
    def __init__(self):
        super(FreqDiscriminator, self).__init__()
        windows = [1024, 768, 720]
        hops = [256, 192, 180]
        n_ffts = [1024, 768, 720]
        self.model = nn.ModuleDict()
        for i, (w, h, n) in enumerate(zip(windows, hops, n_ffts)):
            self.model[f"disc_{i}"] = nn.Sequential(
                STFT(n, h, w),
                DBlock()
                )
        self.apply(weights_init)

    def forward(self, x):
        results = []
        for key, disc in self.model.items():
            results.append(disc(x))
        return results

class Discriminator(nn.Module):
    def __init__(self, num_D, ndf, n_layers, downsampling_factor):
        super(Discriminator, self).__init__()
        self.tdisc = TimeDiscriminator(num_D, ndf, n_layers, downsampling_factor)
        self.fdisc = FreqDiscriminator()
    
    def forward(self, x):
        dt = self.tdisc(x)
        df = self.fdisc(x)
        y = dt + df
        return y
        

if __name__ == "__main__":
    b = 2
    x = torch.randn(2, 1, 8192)    
    #dt = TimeDiscriminator(3, 16, 4, 4)
    #df = FreqDiscriminator()
    d = Discriminator(3, 16, 4, 4)
    y = d(x)
    #x1 = dt(x)
    #x2 = df(x)
    print(x1)
    print(x2)