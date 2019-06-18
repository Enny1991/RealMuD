# function for training and validation
import time
from torch.autograd import Variable
import torch
import sdr
import numpy as np

CONST = 10 * np.log10(np.exp(1))
MSE = torch.nn.MSELoss()

def sSDR(output, clean):
    k = torch.mean(sdr.calc_sdr_torch(output, clean))
    return k


def train(model, train_loader, optimizer, epoch, args):
    start_time = time.time()
    model.train()
    train_loss = 0.
    mixture_loss = 0.

    for batch_idx, data in enumerate(train_loader):
        batch_infeat = Variable(data[0]).contiguous()
        batch_s1 = Variable(data[1][:, 0]).contiguous()  # only ref channel

        if args.cuda:
            batch_infeat = batch_infeat.cuda()
            batch_s1 = batch_s1.cuda()

        optimizer.zero_grad()
        # SDR as objective
        clean = batch_s1
        recon = model(batch_infeat)
        loss = - sSDR(recon, clean)
        show_loss = loss
        total_loss = loss
        # print(-show_loss.data.item() * CONST)
        total_loss.backward()
        train_loss += show_loss.data.item() * CONST
        optimizer.step()

        # print(loss.data.item() * CONST, SDR((s1+s2).unsqueeze(1), batch_infeat.unsqueeze(1), args.cuda).data.item()* CONST)

        if (batch_idx + 1) % args.log_step == 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | clean SDR {:5.4f} |'.format(
                epoch, batch_idx + 1, len(train_loader),
                       elapsed * 1000 / (batch_idx + 1), -train_loss / (batch_idx + 1)))

    train_loss /= (batch_idx + 1)
    print('-' * 99)
    print('    | end of training epoch {:3d} | time: {:5.2f}s | clean SDR {:5.4f} |'.format(
        epoch, (time.time() - start_time), -train_loss))

    return train_loss


def test(model, validation_loader, epoch, args):
    start_time = time.time()
    model.eval()
    validation_loss = 0.
    mixture_loss = 0.

    for batch_idx, data in enumerate(validation_loader):
        batch_infeat = Variable(data[0]).contiguous()
        batch_s1 = Variable(data[1][:, 0]).contiguous()

        if args.cuda:
            batch_infeat = batch_infeat.cuda()
            batch_s1 = batch_s1.cuda()

        with torch.no_grad():
            # SDR as objective
            clean = batch_s1
            recon = model(batch_infeat)
            loss = - sSDR(recon, clean)

        validation_loss += loss.data.item() * CONST

    validation_loss /= (batch_idx + 1)
    # mixture_loss /= (batch_idx+1)
    print('    | end of validation epoch {:3d} | time: {:5.2f}s | clean SDR {:5.4f} |'.format(
        epoch, (time.time() - start_time), -validation_loss))
    print('-' * 99)

    return validation_loss
