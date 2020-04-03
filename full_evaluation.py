import numpy as np
from scipy.io import loadmat
import warnings
from resampy import resample as resample2
import json

from data_utils import norm_power_wav, envelope, MudNoise, unit_power

from sep_eval import sep_eval as se
from model_stft import Mudv4, Mudv5FF, Mudv5LSTM
import os
import torch
import csv
from beamformers import beamformers as bf
from librosa import stft
import pickle as pkl

CONST = 10 * np.log10(np.exp(1))
warnings.simplefilter(action='ignore', category=FutureWarning)

datadir = '/Data/DATASETS/WSJ/mud_noise/'
codedir = '/Data/Dropbox/PhD/Projects/realmud'
noisedir = '/Data/DATASETS/NoiseX/8k/'

base_dir = codedir  # change this!
training_data_path = datadir + 'tr/debug_mud_v1.h5'  # change this!
validation_data_path = datadir + 'cv/debug_mud_v1.h5'  # change this!


def allign(x, y):
    corr = np.correlate(x - np.mean(x), y - np.mean(y), mode='full')
    lag = corr.argmax() - (len(x) - 1)
    _err_log = "OK"

    if lag > 0:
        x = x[lag:]
        y = y[:-lag]
    else:
        x = x[:]
        y = y[:]
        _err_log = "ERROR: lag {}".format(lag)
    return x, y, _err_log


class Evaluator(object):
    def __init__(self):
        
        self.model = None
        self.models = []
        self.models_type = []

        self.measures = []
        self.name_measure = []
        
        self.datasets = []
        self.name_dataset = []
    
    def add_model(self, name, tpe='cnn'):
        self.models.append(name)
        self.models_type.append(tpe)

    def add_measure(self, measure, name=None):
        if name is None:
            name = 'measure{}'.format(len(self.measures))
        self.name_measure.append(name)
        self.measures.append(measure)
    
    def add_dataset(self, mix, s1, n, name=None):
        if name is None:
            name = 'dataset{}'.format(len(self.datasets))
        self.name_dataset.append(name)
        self.datasets.append({'mix': mix, 's1': s1, 'n': n})

    @staticmethod
    def single_test(model, mix):
        # doing batches of 4
        batch_size = 4
        out = []
        for i in range(0, len(mix), batch_size):
            _out = model(mix[i:i + batch_size])
            _out = _out.cpu().data.numpy()
            out.append(_out)
        out = np.vstack(out)
        return out

    @staticmethod
    def single_beam(model, mix, s1, n, mask=None):
        rec = mix.data.numpy()
        y = s1.data.numpy()
        noise = n
        all_y_hat = []

        for _rec, _y, _noise in zip(rec, y, noise):
            if mask is not None:
                y_hat = model(_rec, noise=_noise, target=_y, mask=mask)
            else:
                y_hat = model(_rec, noise=_noise, target=_y)

            # calculate best correlation lag
            all_y_hat.append(y_hat)
        return all_y_hat

    @staticmethod
    def load_mask_model(load, base_dir='.'):
        json_dir = base_dir + '/exp/' + load
        with open(json_dir + '/architecture.json', 'r') as fff:
            p = json.load(fff)
            load_path = json_dir + '/net/' + 'cv/'

            model = Mudv4(n_fft=p['nfft'], kernel=(p['kernel1'], p['kernel2']), causal=p['causal'],
                          layers=p['layers'], stacks=p['stacks'], verbose=False)

            mdl_idx = sorted([int(l.split('_')[-1].split('.')[0]) for l in os.listdir(load_path)])[-1]
            model.load_state_dict(torch.load(load_path + 'model_weight_{}.pt'.format(mdl_idx)))
            _ = model.eval()
            return model, p

    @staticmethod
    def load_lstm_model(load, base_dir='.'):
        json_dir = base_dir + '/exp/' + load
        with open(json_dir + '/architecture.json', 'r') as fff:
            p = json.load(fff)
            load_path = json_dir + '/net/' + 'cv/'

            model = Mudv5LSTM(n_fft=p['nfft'], verbose=False)

            mdl_idx = sorted([int(l.split('_')[-1].split('.')[0]) for l in os.listdir(load_path)])[-1]

            model.load_state_dict(torch.load(load_path + 'model_weight_{}.pt'.format(mdl_idx)))
            _ = model.eval()
            return model, p

    @staticmethod
    def load_blstm_model(load, base_dir='.'):
        json_dir = base_dir + '/exp/' + load
        with open(json_dir + '/architecture.json', 'r') as fff:
            p = json.load(fff)
            load_path = json_dir + '/net/' + 'cv/'

            model = Mudv5LSTM(n_fft=p['nfft'], verbose=False, bidir=True)

            mdl_idx = sorted([int(l.split('_')[-1].split('.')[0]) for l in os.listdir(load_path)])[-1]

            model.load_state_dict(torch.load(load_path + 'model_weight_{}.pt'.format(mdl_idx)))
            _ = model.eval()
            return model, p

    @staticmethod
    def load_ff_model(load, base_dir='.'):
        json_dir = base_dir + '/exp/' + load
        with open(json_dir + '/architecture.json', 'r') as fff:
            p = json.load(fff)
            load_path = json_dir + '/net/' + 'cv/'

            model = Mudv5FF(n_fft=p['nfft'], verbose=False)

            mdl_idx = sorted([int(l.split('_')[-1].split('.')[0]) for l in os.listdir(load_path)])[-1]

            model.load_state_dict(torch.load(load_path + 'model_weight_{}.pt'.format(mdl_idx)))
            _ = model.eval()
            return model, p

    def evaluate(self, file_path='./out.csv', mode='w'):
        
        with open(file_path, mode) as csv_file:
            file_writer = csv.writer(csv_file, delimiter=',')
            # header = ['model', 'dataset', "nfft", "hop", "kernel1", "kernel2", "stacks", "nspk", "layers", "causal", "spkprob", "tr_len", "te_len", "max_len", "noise_train", "frame_train", "noise_test", "frame_test", "dataset"] + self.name_measure
            header = ['model', 'dataset'] + self.name_measure
            file_writer.writerow(header)
            for model, tpe in zip(self.models, self.models_type):
                model_row = [str(model)]

                net = False
                if type(model) == str:
                    print("Loading {}".format(model))
                    if tpe == 'cnn':
                        _model, _p = self.load_mask_model(model)
                    if tpe == 'lstm':
                        _model, _p = self.load_lstm_model(model)
                    if tpe == 'blstm':
                        _model, _p = self.load_blstm_model(model)
                    if tpe == 'ff':
                        _model, _p = self.load_ff_model(model)
                    net = True
                else:
                    _model = model
                # for k, v in _p.items():
                #     model_row.append(v)

                for dataset, name_dataset in zip(self.datasets, self.name_dataset):
                    dataset_row = [name_dataset]
                    if net:
                        _out = self.single_test(_model, dataset['mix'])

                    else:
                        if 'MB_MVDR' in str(model):
                            _out = self.single_beam(_model, dataset['mix'], dataset['s1'], dataset['n'], mask='IRM')
                        else:
                            _out = self.single_beam(_model, dataset['mix'], dataset['s1'], dataset['n'])

                    _s1 = dataset['s1'][:, 0].cpu().data.numpy()

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


def main():
    print("Loading data...")
    # data = pkl.load(open('session_2_enh.pkl', 'rb'))
    
    # test data
    K = MudNoise(validation_data_path, noisedir=noisedir, task='cv', n_ch=6, verbose=True)

    A, B, C = [], [], []
    for i in range(200):
        a, b, c = K[i]
        A.append(a.unsqueeze(0))
        B.append(b.unsqueeze(0))
        C.append(c)

    _mix = torch.cat(A, 0)
    _s1 = torch.cat(B, 0)
    _n = np.array(C)

    print("Verbose test MIX: {}".format(_mix.shape))
    print("Verbose test S1: {}".format(_s1.shape))
    print("Verbose test noise: {}".format(_n.shape))

    # ACTUAL COMPUTE

    evaluator = Evaluator()

    # evaluator.add_model(bf.MB_MVDR_oracle)
    #
    # evaluator.add_model(bf.SDW_MWF)

    # evaluator.add_model('201903015559_Mudv4_NC_256_4')
    #
    # evaluator.add_model('201903050417_Mudv4_NC_256_6')
    #
    # evaluator.add_model('201903072009_Mudv4_C_256_4')
    #
    # evaluator.add_model('201903072324_Mudv4_C_256_6')
    #
    # evaluator.add_model('201903150615_Mudv4_256_NC')

    # evaluator.add_model('201903152712_Mudv4_C_256_4')

    # evaluator.add_model('201903152912_Mudv4_C_512_4')

    # evaluator.add_model('201903152912_Mudv4_NC_512_4')

    # evaluator.add_model('201903152912_Mudv4_NC_512_6')

    # evaluator.add_model('201903152913_Mudv4_C_512_6')

    evaluator.add_model('201906184113_lstm_likeHay_1_NC', 'lstm')
    evaluator.add_model('201906235713_bidirLSTM_likeKHey_2_NC', 'blstm')

    evaluator.add_model('201906002742_ff_likeHey_3_NC', 'ff')

    evaluator.add_dataset(_mix,  _s1, _n, name='verbose test')

    evaluator.add_measure(se.sdr, name='sdr')
    evaluator.add_measure(se.stoi, name='stoi')
    evaluator.evaluate(file_path='./results_sym_data_lstm_dnn.csv', mode='w')
    

if __name__ == "__main__":
    main()
