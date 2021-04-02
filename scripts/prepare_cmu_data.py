import sys, glob, os
from scipy.io.wavfile import write
from librosa.core import load

if __name__ == "__main__":
    speakers = ['rms', 'ahw' ,'jmk' ,'bdl', 'aew','fem']
    for spk in speakers:
    #spk = 'ahe'
        root = '/home/avig/data/speech/cmu_arctic/cmu_us_'+spk+'_arctic/wav'
        rootres = '/home/avig/data/speech/cmu_arctic_2convert'
        fnames = glob.glob(root + '/*.wav')
        for i, f in enumerate(fnames):
            if i % 50 == 0:
                print('{}/{}'.format(i, len(fnames)))
            b = os.path.basename(f)
            ff = rootres + '/' + b[:-4] + '_'+spk+'.wav'
            x, fs = load(f)
            write(ff, rate=fs, data=x)