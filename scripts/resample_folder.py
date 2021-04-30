import glob
import os
import numpy as np
import scipy.io.wavfile as wavfile
import librosa
import argparse
from scipy.io.wavfile import write
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default='../datasets/vcc2018', type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()    
    fnames = glob.glob(args.data_path + '/**/**/*.wav')
    for i, f in enumerate(fnames): 
        if i % 100 == 0:
            print('{}/{}'.format(i, len(fnames)))      
        x, fs = librosa.core.load(f)        
        if fs != 16000:
            x = librosa.core.resample(x, fs, 16000)
        write(f, 16000, (2**15*x).astype(np.int16))