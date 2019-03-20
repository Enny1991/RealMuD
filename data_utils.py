import os
import random
import time

import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

import h5py
import numpy as np
import gc
import soundfile as sf

from scipy.signal import resample_poly as resample
from scipy.io import wavfile

gc.disable()


def norm_power(sig):
    sig -= np.mean(sig, -1, keepdims=True)
    power = np.std(sig, -1, keepdims=True)
    sig /= power
    return sig


def envelope(x, frame_step=125, noise=0.0, compression=0.3):
    x = np.hstack([np.zeros((frame_step,)), x])

    low_pass = np.abs(x) ** compression
    low_pass = resample_poly(low_pass, 1, frame_step)

    return low_pass


def load_wav(filename, max_len=None, fsample=8000):
    old_fs, w = wavfile.read(filename)
    if len(w.shape) > 1:
        w = w[:, 0]  # only channel 0
    w = w.astype('float32')

    if max_len is not None:
        # f_ratio = old_fs / FSAMPLE
        # new_len = max_len * f_ratio
        if max_len < w.shape[0]:
            i = np.random.randint(0, len(w) - max_len - 1)
            w = w[i:i + max_len + 1]

    if old_fs != fsample:
        if old_fs > fsample:
            factor = old_fs // fsample
            w = resample(w, 1, factor)
        else:
            factor = fsample // old_fs
            w = resample(w, factor, 1)

    # return norm_power_wav(w)
    return true_SNR(w)


def true_SNR(sig):
    snr = np.random.rand() * 5. - 2.5
    # first I zero mean
    sig -= np.mean(sig)
    # unit power
    sig /= np.sqrt(np.mean(sig ** 2.)) + 1e-12
    factor = np.sqrt(10 ** (snr / 10.))
    sig *= factor
    if(np.isnan(np.sum(sig))):
        print("!!!nan in true SNR!!!")
    return sig


def unit_power(sig):
    sig -= np.mean(sig)
    # unit power
    sig /= np.sqrt(np.mean(sig ** 2.)) + 1e-12
    if(np.isnan(np.sum(sig))):
        print("!!!nan in unit power!!!")
    return sig


def norm_power_wav(sig):
    sig -= np.mean(sig)
    power = np.max(np.abs(sig))
    sig /= power + 1e-12
    if(np.isnan(np.sum(sig))):
        print("!!!nan in norm wav!!!")
    return sig


def pad_sequences(sequences, dtype='float32', padding='pre', truncating='pre', value=0.):
    max_len = np.max([len(item) for item in sequences])
    # (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)
    nb_samples = len(sequences)
    x = (np.ones((nb_samples, max_len)) * value).astype(dtype)
    mask = (np.ones((nb_samples, max_len)) * value).astype('bool')
    # Check for 2D Audio Histograms or 4D Videos

    # if len(sequences[0].shape) == 2:
    #     x = (np.ones((nb_samples, max_len, sequences[0].shape[1])) * value).astype(dtype)
    #     mask = (np.ones((nb_samples, max_len)) * value).astype('bool')
    # elif len(sequences[0].shape) == 4:
    #     x = (np.ones((nb_samples, max_len,
    #                   sequences[0].shape[1], sequences[0].shape[2], sequences[0].shape[3])) * value).astype(dtype)
    #     mask = (np.ones((nb_samples, max_len)) * value).astype('bool')
    # Loop through and pad
    for idx, s in enumerate(sequences):
        if truncating == 'pre':
            trunc = s[-max_len:]
        elif truncating == 'post':
            trunc = s[:max_len]
        if padding == 'post':
            x[idx, :len(trunc)] = trunc
            mask[idx, :len(trunc)] = 1
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
            mask[idx, -len(trunc):] = 1
    return x, mask


def norm_mc_noise(x, noise, verbose):
    ret = np.zeros_like(x)
    snr = np.random.rand() * 7.5 - 5.0
    if verbose:
        print("SNR: %.2f" % snr)
    pow_noise = np.mean((noise - np.mean(noise)) ** 2.)
    noise /= np.sqrt(pow_noise) + 1e-12
    for i in range(x.shape[0]):
        sig = x[i]
        sig -= np.mean(sig)
        pow_sig = np.mean(sig ** 2.)

        factor = np.sqrt(pow_sig / (10 ** (snr / 10.)))
        ret[i] = x[i] + noise * factor
    return ret


class MudNoise(Dataset):
    def __init__(self, path, noisedir, task='tr', seed=42, max_len=32000, verbose=False,n_ch=6):
        super(MudNoise, self).__init__()

        random.seed(seed)
        np.random.seed(seed)

        self.task = task
        self.max_len = max_len

        data = h5py.File(path, 'r')

        if task == 'tr':
            self.data = [data['{}{}'.format(task, j)] for j in range(10)]
        else:
            self.data = [data['{}{}'.format(task, j)] for j in range(5)]
        # self.len = 5000 if task == 'tr' else 1000
        # self.len = 20000 if task == 'tr' else 5000
        self.len = 1000 if task == 'tr' else 1000

        noises = []
        wav_files = [f for f in os.listdir(noisedir) if 'wav' in f]
        for name in wav_files:
            s, _ = sf.read(noisedir + '/' + name)
            noises.append(s / np.max(np.abs(s)))

        self.noises = noises
        self.verbose = verbose
        self.n_ch = n_ch

    def __getitem__(self, index):
        divider = 2000 if self.task == 'tr' else 1000
        subset = index // divider
        ii = index % divider
        # s = self.data[self.task + "{}".format(subset)][ii][:self.n_ch, :self.max_len]

        s = self.data[subset][ii][:self.n_ch, :self.max_len]
        while np.mean(np.abs(s[0])) == 0:
            index = np.random.choice(self.len)
            subset = index // divider
            ii = index % divider
            # s = self.data[self.task + "{}".format(subset)][ii]
            s = self.data[subset][ii][:self.n_ch, :self.max_len]

        s /= np.max(np.abs(s)) + 1e-15
        idx_noise = np.random.choice(range(len(self.noises)))
        # offset = np.random.choice(range(len(self.noises[idx_noise]) - s.shape[1] - 1))
        #
        offset = np.random.randint(10, len(self.noises[idx_noise]) - s.shape[1] - 1)
        mix = norm_mc_noise(s, self.noises[idx_noise][offset:offset + s.shape[1]], self.verbose)

        mix_tensor = torch.from_numpy(mix.astype('float32'))
        s1_tensor = torch.from_numpy(s.astype('float32'))
        return mix_tensor, s1_tensor

    def __len__(self):
        return self.len
