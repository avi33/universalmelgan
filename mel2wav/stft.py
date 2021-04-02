import torch
import torch.nn as nn
from librosa.filters import mel as librosa_mel_fn
import numpy as np
import torch.nn.functional as F


class STFT(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        sampling_rate=22050,
    ):
        super(STFT, self).__init__()
        ##############################################
        # FFT Parameters                             #
        ##############################################
        window = torch.hann_window(win_length).float()
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate        

    def forward(self, audio):
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audio, (p, p), "reflect").squeeze(1)
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
        )
        real_part, imag_part = fft.unbind(-1)
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)        
        return magnitude

class Audio2Mel(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        sampling_rate=22050,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=None,
    ):
        super(Audio2Mel, self).__init__()
        ##############################################
        # FFT Parameters                              #
        ##############################################        
        mel_basis = librosa_mel_fn(
            sampling_rate, n_fft, n_mel_channels, mel_fmin, mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()        
        self.register_buffer("mel_basis", mel_basis)
        self.stft = STFT(n_fft, hop_length, win_length, sampling_rate)     
        self.n_mel_channels = n_mel_channels

    def forward(self, x):
        mag = self.stft(x)        
        mel_output = torch.matmul(self.mel_basis, mag)
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
        return log_mel_spec


if __name__ == "__main__":
    x = torch.randn(2, 1, 8192)
    fft = STFT()
    X = fft(x)
    print(X.shape)