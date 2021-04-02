import os, sys, glob
import numpy as np
import librosa

def copyWavFiles():
    root = "/home/avig/data/speech/cmu_arctic/"
    root_target = "/home/avig/data/speech/cmu_arctic_all"
    files = glob.glob(root + '/**/**/*.wav')
    print("found {} files".format(len(files)))
    fs = 22050
    for i, f in enumerate(files):
        if i % 50 == 0:
            print("{}/{}".format(i, len(files)))
        b = os.path.basename(f)
        ff = root_target + '/' + b
        x, _ = librosa.core.load(f, sr=fs)
        librosa.output.write_wav(ff, x, fs)

def splitTrainVal(root_target, speakers):
    files = glob.glob(root_target + '/*.wav')
    files_spks = []
    for f in files:
        spk = os.path.basename(f).split('_')[-1][:-4]
        if spk in speakers:
            files_spks.append(f)
    rand_idx = np.random.permutation(len(files_spks))
    train_idx = rand_idx[:-10]
    val_idx = rand_idx[-10:]
    with open(root_target + "/train_files.txt", 'w') as f:  
        for i in range(train_idx.shape[0]): 
            b = os.path.basename(files_spks[train_idx[i]])+'\n'
            f.write(b)
    with open(root_target + "/test_files.txt", 'w') as f:  
        for i in range(val_idx.shape[0]): 
            b = os.path.basename(files_spks[val_idx[i]])+'\n'
            f.write(b)             
    

if __name__ == "__main__":
    #root_target = "/home/avig/data/speech/cmu_arctic"
    #root_target = "/home/avig/data/speech/cmu_arctic/cmu_us_ahw_arctic/wav"
    #root_target = "/home/avig/data/speech/DS_10283_2651/VCTK-Corpus/wav22-5/p226"    
    #copyWavFiles(root_target)
    speakers = ['aew', 'bdl', 'jmk', 'ahw', 'fem', 'rms']
    #speakers = ['aew', 'rms']
    root_target = '/home/avig/data/speech/cmu_arctic_2convert'
    splitTrainVal(root_target, speakers)