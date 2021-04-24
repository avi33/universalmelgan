import os, glob
import numpy as np
import argparse

def splitTrainVal(root, fnames):
    rand_idx = np.random.permutation(len(fnames))
    train_idx = rand_idx[:-10]
    val_idx = rand_idx[-10:]
    with open(root + "/train_files_inv.txt", 'w') as f:  
        for i in range(train_idx.shape[0]): 
            b = fnames[train_idx[i]]+'\n'
            f.write(b)
    with open(root + "/test_files_inv.txt", 'w') as f:  
        for i in range(val_idx.shape[0]): 
            b = fnames[val_idx[i]]+'\n'
            f.write(b)             
    
def parse_args():
    parser = argparse.ArgumentParser()      
    parser.add_argument("--data_path", default='/media/avi/8E56B6E056B6C86B/datasets', type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()    
    audio_sets = ['ARCTIC','vcc2018','vctk', 'bibi', 'zamir']    
    fnames = []
    for i, audio_set in enumerate(audio_sets):
        if audio_set.lower() == 'arctic':
            ff = glob.glob(os.path.join(args.data_path, audio_set) + '/**/wav/*.wav')            
            fnames += ff
        elif audio_set.lower() == 'vctk':
            ff = glob.glob(os.path.join(args.data_path, audio_set, 'clean_trainset_28spk_wav') + '/*.wav')
            fnames += ff
        elif audio_set.lower() == 'vcc2018':
            ff = glob.glob(os.path.join(args.data_path, 'vcc2018') + '/**/**/*.wav')
            fnames += ff
        elif audio_set.lower() == 'bibi':
            ff = glob.glob(os.path.join(args.data_path, audio_set) + '/*.wav')
            fnames += ff
        elif audio_set.lower() == 'zamir':
            ff = glob.glob(os.path.join(args.data_path, audio_set) + '/*.wav')
            fnames += ff
        else:
            raise NotImplemented
        print("set:{} #files:{}".format(audio_set.lower(), len(ff)))
    
    splitTrainVal(args.data_path, fnames)