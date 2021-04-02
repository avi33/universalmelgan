import os
import numpy as np
import torch
from scipy.io.wavfile import write

def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding="utf-8") as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    with torch.no_grad():
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()    
        correct = pred.eq(target.view(1, -1).expand_as(pred))    
    return [correct[:k].view(-1).float().sum(0) * 100. / batch_size for k in topk]


def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    From Keras np_utils
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return torch.Tensor(categorical)


def sample_spk_c(sz, num_speakers):
    spk_c = np.random.randint(0, num_speakers, size=sz)
    spk_c_cat = to_categorical(spk_c, num_speakers)
    return torch.LongTensor(spk_c), torch.FloatTensor(spk_c_cat)


def to_onehot(idx, n_classes):
    idx = idx % n_classes
    onehot = torch.zeros(len(idx), n_classes).scatter_(1, idx.unsqueeze(1), 1.)
    return onehot


def save_sample(file_path, sampling_rate, audio):
    """Helper function to save sample

    Args:
        file_path (str or pathlib.Path): save file path
        sampling_rate (int): sampling rate of audio (usually 22050)
        audio (torch.FloatTensor): torch array containing audio in [-1, 1]
    """
    audio = (audio.numpy() * 32768).astype("int16")
    write(file_path, sampling_rate, audio)


def rms(x):
    r = np.sqrt((x**2).mean())
    return r
