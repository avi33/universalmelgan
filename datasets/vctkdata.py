import torch
import torch.nn.functional as F
import torchaudio
from pathlib import Path
import glob
import os
from librosa.core import load
import random

def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding="utf-8") as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files

class VCTK(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_length, sampling_rate, noisy=True):        
        files = files_to_list(training_files)
        self.use_noisy_paired = noisy    
        self.fnames = [Path(training_files).parent / Path('clean_trainset_28spk_wav') / x for x in files]
        if self.use_noisy_paired:
            self.fnames_noisy = [Path(training_files).parent / Path('noisy_trainset_28spk_wav') / x for x in files]
        self.speakers = self._get_spks()        
        self.num_speakers = len(self.speakers)                        
        self.spk_idx = dict(zip(self.speakers, range(self.num_speakers)))
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length        
            
    
    def __getitem__(self, index):
        filename = self.fnames[index]
        spk = os.path.basename(filename).split('_')[0]
        spk_idx = self.spk_idx[spk]

        if self.use_noisy_paired:
            filename_noisy = self.fnames_noisy[index]
            audio, audio_noisy = self.get_sample_pairs(filename, filename_noisy)
            return audio.unsqueeze(0), audio_noisy.unsqueeze(0), spk_idx
        else:        
            audio = self.get_sample(filename)
            return audio.unsqueeze(0), spk_idx
            
    def __len__(self) -> int:
        return len(self.fnames)

    def get_sample(self, filename):
        audio, _ = load(filename, sr=self.sampling_rate)
        audio = torch.from_numpy(audio).float()
        if audio.size(0) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start : audio_start + self.segment_length]            
        else:
            audio = F.pad(
                audio, (0, self.segment_length - audio.size(0)), "constant"
            ).data            
        return audio

    def get_sample_pairs(self, filename, filename_noisy):        
        audio, _ = load(filename, sr=self.sampling_rate)
        audio_noisy, _ = load(filename_noisy, sr=self.sampling_rate)
        audio = torch.from_numpy(audio).float()
        audio_noisy = torch.from_numpy(audio_noisy).float()
        if audio.size(0) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start : audio_start + self.segment_length]
            audio_noisy = audio_noisy[audio_start : audio_start + self.segment_length]
        else:
            audio = F.pad(
                audio, (0, self.segment_length - audio.size(0)), "constant"
            ).data
            audio_noisy = F.pad(
                audio_noisy, (0, self.segment_length - audio_noisy.size(0)), "constant"
            ).data
        return audio, audio_noisy

    def _get_spks(self):
        speakers = []
        for f in self.fnames:
            name = os.path.basename(f)
            ff = name.split('_')[0]
            speakers.append(ff)
        speakers = list(set(speakers))
        return speakers

if __name__ == "__main__":
    files = Path('/home/avi/Documents/projects/speech/vctk') / "train_files.txt"
    D = VCTK(files, 6400)
    next(iter(D))