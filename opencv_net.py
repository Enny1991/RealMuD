import numpy as np
import time
from model_stft import Mudv3noFFT
import json
import os
import torch
from torch import nn
from torch.autograd import Variable
import csv
from tqdm import tqdm


def load_mask_model(load, base_dir='./'):
    json_dir = base_dir + '/exp/' + load
    with open(json_dir + '/architecture.json', 'r') as fff:
        p = json.load(fff)
        load_path = json_dir + '/net/' + 'cv/'

        model = Mudv3noFFT(n_fft=p['nfft'], kernel=(p['kernel1'], p['kernel2']), causal=p['causal'],
                                layers=p['layers'], stacks=p['stacks'], verbose=False)

        mdl_idx = sorted([int(l.split('_')[-1].split('.')[0]) for l in os.listdir(load_path)])[-1]
        model.load_state_dict(torch.load(load_path + 'model_weight_{}.pt'.format(mdl_idx)))
        _ = model.eval()
        return model, p



mdl, p = load_mask_model('201903161143_512_4l_NC')
print(p)
re_m = torch.from_numpy(np.random.randn(1, 2, 129, 128).astype('float32'))
im_m = torch.from_numpy(np.random.randn(1, 2, 129, 128).astype('float32'))

mix = torch.cat([re_m, im_m], -1)
mix = Variable(mix).contiguous()

curr = []

for i in tqdm(range(20)):
    start_time = time.time()
    recon = mdl(mix)
    curr.append(time.time() - start_time)


print("Took {} +/- {} s to process".format(np.mean(curr), np.std(curr)))
