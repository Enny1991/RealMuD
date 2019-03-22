import numpy as np
from scipy.io import loadmat
import warnings
from resampy import resample as resample2
import json

from data_utils import norm_power_wav, envelope, MudNoise, unit_power

from sep_eval import sep_eval as se
from model_stft import Mudv3
import os
import torch
import csv
from librosa import stft

CONST = 10 * np.log10(np.exp(1))
warnings.simplefilter(action='ignore', category=FutureWarning)


class Evaluator(object):
    def __init__(self):
        
        self.model = None
        self.models = []
        
        self.measures = []
        self.name_measure = []
        
        self.datasets = []
        self.name_dataset = []
    
    def add_model(self, name):
        self.models.append(name)
    
    def add_measure(self, measure, name=None):
        if name is None:
            name = 'measure{}'.format(len(self.measures))
        self.name_measure.append(name)
        self.measures.append(measure)
    
    def add_dataset(self, mix, env, s1, name=None):
        if name is None:
            name = 'model{}'.format(len(self.models))
        self.name_dataset.append(name)
        self.datasets.append({'mix': mix, 'env': env, 's1': s1})

    @staticmethod
    def single_test(model, mix, env):
        # doing batches of 4
        batch_size = 4
        out = []
        for i in range(0, len(mix), batch_size):
            _out = model([mix[i:i + batch_size], env[i:i + batch_size]])
            _out = _out.cpu().data.numpy()
            out.append(_out)
        out = np.vstack(out)
        return out

    @staticmethod
    def load_mask_model(load, base_dir='.'):
        json_dir = base_dir + '/exp/' + load
        with open(json_dir + '/architecture.json', 'r') as fff:
            p = json.load(fff)
            load_path = json_dir + '/net/' + 'cv/'

            model = Mudv3(n_fft=p['nfft'], kernel=(p['kernel1'], p['kernel2']), causal=p['causal'],
                          layers=p['layers'], stacks=p['stacks'], verbose=False)

            mdl_idx = sorted([int(l.split('_')[-1].split('.')[0]) for l in os.listdir(load_path)])[-1]
            model.load_state_dict(torch.load(load_path + 'model_weight_{}.pt'.format(mdl_idx)))
            _ = model.eval()
            return model, p

    def evaluate(self, file_path='./out.csv', mode='w'):
        
        with open(file_path, mode) as csv_file:
            file_writer = csv.writer(csv_file, delimiter=',')
            header = ['model', 'dataset', "nfft", "hop", "kernel1", "kernel2", "stacks", "nspk", "layers", "causal", "spkprob", "tr_len", "te_len", "max_len", "noise_train", "frame_train", "noise_test", "frame_test", "dataset"] + self.name_measure
            file_writer.writerow(header)
            for model in self.models:
                model_row = [model]
                
                print("Loading {}".format(model))
                _model, _p = self.load_mask_model(model)

                for k, v in _p.items():
                    model_row.append(v)

                for dataset, name_dataset in zip(self.datasets, self.name_dataset):
                    dataset_row = [name_dataset]
                    _out = self.single_test(_model, dataset['mix'], dataset['env'])
                    _s1 = dataset['s1'].cpu().data.numpy()
                    collected_measures = []
                    for measure, name_measure in zip(self.measures, self.name_measure):
                        collected_measures.append(measure(_out, _s1))
                    # separate
                    res = np.zeros((len(self.measures), len(collected_measures[0])))
                    for j, m in enumerate(collected_measures):
                        for i, val in enumerate(m):
                            res[j, i] = val
                    for i in range(res.shape[1]):
                        _t = []
                        for j in range(res.shape[0]):
                            _t.append(res[j, i])
                        file_writer.writerow(model_row + dataset_row + _t)
                del _model  # for RAM


def find_trigger_v2(x):
    xx = stft(x)
    y = np.log10(np.abs(xx) ** 2 + 1e-8)
    yy = np.sum(y, 0)
    n90 = np.percentile(yy, 65)
    tr = np.where(yy >= n90)[0][0]
    return tr * 512


def main():
    print("Loading data...")
    data = loadmat('data_attended.mat')
    mat_env = loadmat('data_attended_env.mat')

    _y = resample2(data['a']['b1_attend'][0][0].squeeze(), 16000, 8000)
    _y2 = resample2(data['a']['b2_attend'][0][0].squeeze(), 16000, 8000)

    min_len = min([len(_y), len(_y2)])
    _y = _y[:min_len]
    _y2 = _y2[:min_len]
    # random mixing of the two voices with 0 dB
    mx = unit_power(_y[:min_len]) + unit_power(_y2)
    mx = norm_power_wav(mx)

    print("Have a test file of {:.4} s".format(mx.shape[0] / 8000.))
    
    # set some params
    fs = 8000
    len_sample = fs * 4
    rate_env = 64
    shift = int(fs / rate_env)
    l = int(len_sample / shift)
    
    # arrange data
    y1 = _y[:-int(len(_y) % len_sample)]
    y2 = _y2[:-int(len(_y2) % len_sample)]
    mx1 = mx[:-int(len(mx) % len_sample)]

    y1 = np.reshape(y1, (-1, len_sample))
    y2 = np.reshape(y2, (-1, len_sample))
    mx1 = np.reshape(mx1, (-1, len_sample))

    print("SHape of data: {}".format(mx1.shape))

    raw_gt1 = mat_env['env1'].squeeze()
    raw_gt1 = resample2(raw_gt1, 100, rate_env)

    raw_gt1 = raw_gt1[:-int(len(raw_gt1) % l)]
    raw_gt1 = np.reshape(raw_gt1, (-1, l))

    if rate_env == 64:
        raw_gt1 = np.concatenate([np.zeros((raw_gt1.shape[0], 1)) ,raw_gt1], 1)
    else:
        raw_gt1 = np.concatenate([np.zeros((raw_gt1.shape[0], 1)) ,raw_gt1, np.zeros((raw_gt1.shape[0], 1))], 1)

    print("Shape of estimated env1: {}".format(raw_gt1.shape))

    raw_gt2 = mat_env['env2'].squeeze()
    raw_gt2 = resample2(raw_gt2, 100, rate_env)

    raw_gt2 = raw_gt2[:-int(len(raw_gt2) % l)]
    raw_gt2 = np.reshape(raw_gt2, (-1, l))
    if rate_env == 64:
        raw_gt2 = np.concatenate([np.zeros((raw_gt2.shape[0], 1)), raw_gt2], 1)
    else:
        raw_gt2 = np.concatenate([np.zeros((raw_gt2.shape[0], 1)), raw_gt2, np.zeros((raw_gt2.shape[0], 1))], 1)

    print("Shape of estimated env2: {}".format(raw_gt2.shape))

    real_gt1 = np.array([envelope(y, frame_step=shift) for y in y1])
    real_gt2 = np.array([envelope(y, frame_step=shift) for y in y2])

    print("Shape of real env1: {}".format(real_gt2.shape))
    print("Shape of real env2: {}".format(real_gt2.shape))

    mix = torch.from_numpy(mx1.astype('float32'))
    s1 = torch.from_numpy(y1.astype('float32'))
    s2 = torch.from_numpy(y2.astype('float32'))
    raw_gt1 = torch.from_numpy(raw_gt1.astype('float32'))
    raw_gt2 = torch.from_numpy(raw_gt2.astype('float32'))
    real_gt1 = torch.from_numpy(real_gt1.astype('float32'))
    real_gt2 = torch.from_numpy(real_gt2.astype('float32'))
    
    # test data
    K = WSJDatasetSTFTv3('/work3/jhjort/biss/danish_english_singles_v3.h5', task='valid', max_len=32000)

    A, B, C = [], [], []
    for i in range(200):
        a, b, c = K[i]
        A.append(a.unsqueeze(0))
        B.append(b.unsqueeze(0))
        C.append(c.unsqueeze(0))
    _mix = torch.cat(A, 0)
    _s1 = torch.cat(B, 0)
    _real_gt1 = torch.cat(C, 0)

    print("Verbose test MIX: {}".format(_mix.shape))
    print("Verbose test S1: {}".format(_s1.shape))
    print("Verbose test ENV1: {}".format(_real_gt1.shape))

    print("REAL test MIX: {}".format(mix.shape))
    print("REAL test S1: {}".format(s1.shape))
    print("REAL test ENV1: {}".format(real_gt1.shape))
    
    # ACTUAL COMPUTE

    evaluator = Evaluator()
    
    evaluator.add_model('201903093515_DEv4_corr8_fft_512_hop_125_C_0_kT_3_kF_3_S_2_L_6_NO_0.0_FR_1')

    evaluator.add_dataset(_mix, _real_gt1, _s1, name='verbose test')
    evaluator.add_dataset(mix, real_gt1, s1, name='real test 1')
    evaluator.add_dataset(mix, real_gt2, s2, name='real test 2')

    evaluator.add_measure(se.sdr, name='sdr')
    evaluator.add_measure(se.stoi, name='stoi')
    evaluator.evaluate()
    

if __name__ == "__main__":
    main()
