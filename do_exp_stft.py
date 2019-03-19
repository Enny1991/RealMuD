import argparse
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import json
import datetime
import os
import numpy as np
import data_utils
from logger import Logger
import getpass

# prepare exp folder
from model_stft import Mud, Mudv2, Mudv3
from train_test import train, test

if getpass.getuser() == 'enea':
    datadir = '/Data/DATASETS/WSJ/mud_noise/'
    codedir = '/Data/Dropbox/PhD/Projects/realmud'
elif getpass.getuser() == 'jhjort':
    datadir = '/work3/jhjort/biss/out/'
    codedir = '/work3/jhjort/biss/'
else:
    raise ValueError('unknown user')

base_dir = codedir  # change this!
training_data_path = datadir + 'tr/debug_mud_v1.h5'  # change this!
validation_data_path = datadir + 'cv/debug_mud_v1.h5'  # change this!


def main(args):
    num_gpu = torch.cuda.device_count()
    print("NUM GPUS: {}".format(num_gpu))
    print(args.exp_name)
    # fill more info in the name
    exp_name = datetime.datetime.now().strftime('%G%m%H%M%S_')
    exp_name += args.exp_name
    if args.causal == 1:
        exp_name += '_C'
    else:
        exp_name += '_NC'

    if not os.path.isdir(base_dir + '/exp/' + exp_name):
        os.mkdir(base_dir + '/exp/' + exp_name)
        os.mkdir(base_dir + '/exp/' + exp_name + '/net/')
        os.mkdir(base_dir + '/exp/' + exp_name + '/net/' + '/cv/')

    logdir = base_dir + '/exp/' + exp_name  # tensorboard logger directory
    logger = Logger(logdir)  # tensorboard logger object
    print('Log directory = {}'.format(logdir))
    val_save = base_dir + '/exp/' + exp_name + '/net/cv/model_weight_'

    # global params
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        kwargs = {'num_workers': 1, 'pin_memory': True}
    else:
        kwargs = {}

    # loaders
    # define data loaders
    print("Preparing Loaders...")

    train_loader = DataLoader(
        data_utils.MudNoise(training_data_path,  noisedir='/Data/DATASETS/NoiseX/8k/', task='tr', n_ch=6),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs)

    validation_loader = DataLoader(
        data_utils.MudNoise(validation_data_path, noisedir='/Data/DATASETS/NoiseX/8k/', task='cv', n_ch=6),
        batch_size=args.batch_size,
        shuffle=False,
        **kwargs)

    args.dataset_len = len(train_loader)
    args.log_step = args.dataset_len // 8
    print("Creating Model...")
    # define model

    model = Mudv3(n_fft=args.nfft, hop=args.hop, kernel=(args.kernel1, args.kernel2), causal=args.causal == 1,
                  layers=args.layers, stacks=args.stacks)

    if args.load is not None:
        print("Loading model {}".format(args.load))
        load_path = base_dir + '/exp/' + args.load + '/net/' + 'cv/'
        mdl_idx = sorted([int(l.split('_')[-1].split('.')[0]) for l in os.listdir(load_path)])[-1]
        print("Loading model {}".format(load_path + 'model_weight_{}.pt'.format(mdl_idx)))
        model.load_state_dict(torch.load(load_path + 'model_weight_{}.pt'.format(mdl_idx)))
    # print(model)

    print("Doing cuda")
    model.cuda()
    print("IS CUDA: {}".format(next(model.parameters()).is_cuda))

    s = 0
    for param in model.parameters():
        s += np.product(param.size())
    print('# of parameters: ' + str(s))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    scheduler.step()

    params = {'nfft': args.nfft,
              'hop': args.hop,
              'kernel1': args.kernel1,
              'kernel2': args.kernel2,
              'stacks': args.stacks,
              'nspk': args.nspk,
              'layers': args.layers,
              'causal': args.causal,
              'spkprob': args.spkprob,
              'tr_len': args.tr_len,
              'te_len': args.te_len,
              'max_len': args.max_len,
              'dataset': training_data_path,
              }

    with open(logdir + '/architecture.json', 'w') as fff:
        json.dump(params, fff)

    # description
    description = json.dumps(params)

    logger.text_summary('model', description)
    print("Logged")

    # main
    training_loss = []
    validation_loss = []
    decay_cnt = 0

    print("Starting Training...")

    for epoch in range(1, args.epochs + 1):
        if args.cuda:
            model.cuda()

        training_loss.append(train(model, train_loader, optimizer, epoch, args))
        validation_loss.append(test(model, validation_loader, epoch, args))

        logger.scalar_summary('Train loss over epochs', training_loss[-1], epoch)
        logger.scalar_summary('Test loss over epochs', validation_loss[-1], epoch)

        if training_loss[-1] == np.min(training_loss):
            print('      Best training model found.')
            print('-' * 99)
        if validation_loss[-1] == np.min(validation_loss):
            # save current best model
            with open(val_save + '{}.pt'.format(epoch), 'wb') as f:
                torch.save(model.cpu().state_dict(), f)
                print('      Best validation model found and saved.')
                print('-' * 99)
        decay_cnt += 1

        # lr decay
        if np.min(training_loss) not in training_loss[-3:] and decay_cnt >= 3:
            scheduler.step()
            decay_cnt = 0
            print('      Learning rate decreased.')
            print('-' * 99)

    np.save(base_dir + '/exp/' + exp_name + '/validation_loss', validation_loss)
    np.save(base_dir + '/exp/' + exp_name + '/training_loss', training_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tasnet-enhancement')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--noise', type=float, default=0.0,
                        help='noise envelope')
    parser.add_argument('--seed', type=int, default=20180825,
                        help='random seed')
    # parser.add_argument('--val-save', type=str, default=base_dir + '/exp/' + exp_name + '/net/cv/model_weight_',
    #                    help='path to save the best model')
    parser.add_argument('--nfft', type=int, default=512)
    parser.add_argument('--hop', type=int, default=125)
    parser.add_argument('--kernel1', type=int, default=3)
    parser.add_argument('--kernel2', type=int, default=3)
    parser.add_argument('--stacks', type=int, default=2)
    parser.add_argument('--nspk', type=int, default=2)
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--mics', type=int, default=2)
    parser.add_argument('--causal', type=int, default=0)
    parser.add_argument('--spkprob', type=int, default=0)
    parser.add_argument('--tr_len', type=int, default=20000)
    parser.add_argument('--te_len', type=int, default=5000)
    parser.add_argument('--max_len', type=int, default=32000)

    parser.add_argument('--exp_name', type=str, default='tasnet')
    parser.add_argument('--load', type=str, default=None)
    args, _ = parser.parse_known_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    main(args)
