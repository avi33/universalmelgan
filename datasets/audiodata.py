import os
import torch
import torchaudio
torchaudio.set_audio_backend('sox_io')
import torch.utils.data
import torch.nn.functional as F
from librosa.core import load, resample
from librosa.util import normalize
from pathlib import Path
import numpy as np
import random
from mel2wav.utils import files_to_list
from mel2wav.audio_augs import AudioAugs


class AudioDataset(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, training_files, segment_length, sampling_rate, augment=None):        
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.audio_files = files_to_list(training_files)
        self.audio_files = [training_files.parent / Path(x) for x in self.audio_files]
        self.augs = AudioAugs(fs=sampling_rate, augs=augment) if augment else None

    def __getitem__(self, index):
        filename = self.audio_files[index]
        audio, sampling_rate = self.load_wav_to_torch(filename)
        # Take segment
        if audio.size(0) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start : audio_start + self.segment_length]
        else:
            audio = F.pad(
                audio, (0, self.segment_length - audio.size(0)), "constant"
            ).data            
        return audio.unsqueeze(0)

    def __len__(self):
        return len(self.audio_files)

    def get_speakers(self):
        spks = [os.path.dirname(f).split('/')[-2].split('_')[-2] for f in self.audio_files]
        spks = list(set(spks))        
        return spks


    def load_wav_to_torch(self, full_path):
        """
        Loads wavdata into torch array
        """
        data, sampling_rate = load(full_path, sr=None)        
        data = 0.95 * normalize(data)
        data = torch.from_numpy(data).float()
        if self.augs is not None:
            data = self.augs(data)        
        return data, sampling_rate