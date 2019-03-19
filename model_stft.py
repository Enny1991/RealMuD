import torch
import numpy as np
from torch import nn
from torch.autograd import Variable


def my_istft(stft, hop_length, cuda=False):
    # B, F, T
    batch_size = stft.shape[0]
    n_fft = 2 * (stft.shape[1] - 1)
    ifft_window = torch.hann_window(n_fft)
    if cuda:
        ifft_window = ifft_window.cuda()

    # By default, use the entire frame
    win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    n_frames = stft.shape[2]
    expected_signal_len = n_fft + hop_length * (n_frames - 1)

    y = torch.zeros(batch_size, expected_signal_len, dtype=torch.float32)
    if cuda:
        y = y.cuda()
    ifft_window_sum = torch.zeros(batch_size, expected_signal_len, dtype=torch.float32)
    if cuda:
        ifft_window_sum = ifft_window_sum.cuda()
    ifft_window_square = ifft_window * ifft_window

    for i in range(n_frames):
        sample = i * hop_length
        spec = stft[:, :, i]
        y_tmp = torch.irfft(spec, signal_ndim=1, signal_sizes=(n_fft,))
        if cuda:
            y_tmp = y_tmp.cuda()

        y[:, sample:(sample + n_fft)] = y[:, sample:(sample + n_fft)] + y_tmp
        ifft_window_sum[:, sample:(sample + n_fft)] += ifft_window_square

    y = y[:, int(n_fft // 2):-int(n_fft // 2)] / 2
    return y


def trace(x, dim1=-1, dim2=-2, keepdim=True):
    if x.shape[dim1] != x.shape[dim2]:
        raise ValueError("Matrix should be square")
    n = len(x.shape) - 2
    ones = torch.eye(x.shape[dim1])[(None,) * n]  # [..., dim1, dim2]
    if x.is_cuda:
        ones = ones.cuda()
    filt = ones * x
    return filt.sum((dim1, dim2), keepdim=keepdim)


def condition_covariance(x, gamma):
    """see https://stt.msu.edu/users/mauryaas/Ashwini_JPEN.pdf (2.3)"""
    scale = gamma * trace(x, keepdim=True) / x.shape[-1]  # [...]
    n = len(x.shape) - 2
    eye = torch.eye(x.shape[-1])[(None,) * n]
    if x.is_cuda:
        eye = eye.cuda()
    scaled_eye = (eye * scale)
    return (x + scaled_eye) / (1 + gamma)


def complex_multiply(x, y):
    # x: (B, M, 2, F, T)
    a = x[:, :, 0].unsqueeze(1)
    b = x[:, :, 1].unsqueeze(1)
    c = y[:, :, 0].unsqueeze(1)
    d = y[:, :, 1].unsqueeze(1)

    real = a * c - b * d
    imag = a * d + b * c
    return torch.cat([real, imag], 2)


def einsum(a, b, s='...dt,...et->...de'):
    return torch.einsum(s, a, b)


def complex_psd(x, mask, normalize=True, condition=True, eps=1e-15):
    # x: (B, M, 2, F, T)
    masked = x * mask  # (B, M, 2, F, L)

    a = masked[:, :, 0].unsqueeze(2)  # re x
    b = masked[:, :, 1].unsqueeze(2)  # im x
    c = x[:, :, 0].unsqueeze(2)
    d = - x[:, :, 1].unsqueeze(2)  # im y has been conjugated

    real = einsum(a, c) - einsum(b, d)
    imag = einsum(a, d) + einsum(b, c)
    psd = torch.cat([real, imag], 2)

    if normalize:
        normalization = mask.sum(-1).unsqueeze(-1)
        psd /= normalization + eps

    return psd


class tLN(nn.Module):
    def __init__(self, dimension, eps=1e-8, trainable=True):
        super(tLN, self).__init__()

        self.eps = eps
        if trainable:
            self.gain = nn.Parameter(torch.ones(1, dimension, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, dimension, 1, 1))
        else:
            self.gain = Variable(torch.ones(1, dimension, 1, 1), requires_grad=False)
            self.bias = Variable(torch.zeros(1, dimension, 1, 1), requires_grad=False)

    def forward(self, inp):
        # input size: (Batch, Ch, Freq, Time)

        batch_size = inp.size(0)
        mean = torch.mean(inp, 3, keepdim=True)
        std = torch.sqrt(torch.var(inp, 3, keepdim=True) + self.eps)

        x = (inp - mean.expand_as(inp)) / std.expand_as(inp)
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())


class DepthConv2d(nn.Module):
    def __init__(self, input_channel, hidden_channel, kernel,
                 dilation=(1, 1), stride=(1, 1), padding=(0, 0), causal=False):
        super(DepthConv2d, self).__init__()

        self.padding = padding

        # self.linear = nn.Conv2d(input_channel, hidden_channel, (1, 1), groups=4)
        self.linear = nn.Conv2d(input_channel, hidden_channel, (1, 1))
        if causal:
            self.conv1d = CausalConv2d(hidden_channel, hidden_channel, kernel,
                                       stride=stride,
                                       dilation=dilation)
        else:
            self.conv1d = nn.Conv2d(hidden_channel, hidden_channel, kernel,
                                    stride=stride, padding=self.padding,
                                    dilation=dilation)

        # self.BN = nn.Conv2d(hidden_channel, input_channel, (1, 1), groups=4)
        self.BN = nn.Conv2d(hidden_channel, input_channel, (1, 1))

        self.nonlinearity1 = nn.PReLU()
        self.nonlinearity2 = nn.PReLU()

        self.reg1 = tLN(hidden_channel)
        self.reg2 = tLN(hidden_channel)

    def forward(self, input):
        output = self.reg1(self.nonlinearity1(self.linear(input)))
        output = self.reg2(self.nonlinearity2(self.conv1d(output)))
        output = self.BN(output)

        return output

    
class CausalConv2d(torch.nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=(1, 1),
                 dilation=(1, 1),
                 groups=1,
                 bias=True):
        _pad = (int(np.log2((kernel_size[1] - 1) / 2)))
        padding_2 = int(2 ** (np.log2(dilation[1]) + _pad))
        self.__padding = ((kernel_size[0] - 1) * dilation[0], padding_2)

        super(CausalConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input):
        result = super(CausalConv2d, self).forward(input)
        print("\tconv --")
        if self.__padding[0] != 0:
            return result[:, :, :-self.__padding[0]]
        return result


class Mudv2(nn.Module):
    def __init__(self, n_fft=256, hop=125, learn_comp=False, bn_ch=32, sep_ch=64, kernel=(3, 3), causal=False, layers=6, stacks=2, verbose=True):
        super(Mudv2, self).__init__()
        if verbose:
            print("NFFT IS: {}".format(n_fft))
        self.n_fft = (n_fft // 2 + 1)
        self.FFT = n_fft
        self.HOP = hop
        self.BN_channel = bn_ch
        self.conv_channel = sep_ch
        self.kernel = kernel
        self.conv_pad = (int(np.log2((self.kernel[0] - 1) / 2)), int(np.log2((self.kernel[1] - 1) / 2)))

        self.enc_LN = tLN(2)
        self.BN = nn.Conv2d(2, self.BN_channel, (1, 1))

        self.compression = nn.Parameter(torch.ones(1,) * -0.8475, requires_grad=learn_comp)  # sigmoid(-0.8475) = 0.3
        if verbose:
            print("Compression: {}".format(torch.sigmoid(self.compression)))
        self.layer = layers
        self.stack = stacks

        self.receptive_field_time = 0
        self.receptive_field_freq = 0
        self.conv = nn.ModuleList([])
        for s in range(self.stack):
            for i in range(self.layer):
                self.conv.append(DepthConv2d(self.BN_channel, self.conv_channel,
                                             self.kernel, dilation=(2 ** i, 2 ** i), causal=causal,
                                             padding=(2 ** (i + self.conv_pad[0]), 2 ** (i + self.conv_pad[1]))))
                if s == 0 and i == 0:
                    self.receptive_field_time += self.kernel[0]
                    self.receptive_field_freq += self.kernel[1]
                else:
                    self.receptive_field_time += (self.kernel[0] - 1) * 2 ** i
                    self.receptive_field_freq += (self.kernel[1] - 1) * 2 ** i

        if verbose:
            print('Receptive field TIME: {:1d} samples.'.format(self.receptive_field_time))
            print('Receptive field FREQ: {:1d} samples.'.format(self.receptive_field_freq))

        self.conv2 = nn.ModuleList([])
        for s in range(self.stack):
            for i in range(self.layer):
                self.conv2.append(DepthConv2d(self.BN_channel, self.conv_channel,
                                              self.kernel, dilation=(2 ** i, 2 ** i), causal=causal,
                                              padding=(2 ** (i + self.conv_pad[0]), 2 ** (i + self.conv_pad[1]))))

        self.reshape1 = nn.Sequential(nn.Conv2d(self.BN_channel, 1, (1, 1)), nn.Tanh())

        self.reshape_noise = nn.Sequential(nn.Conv2d(self.BN_channel, 1, (1, 1)), nn.Sigmoid())
        self.reshape_speech = nn.Sequential(nn.Conv2d(self.BN_channel, 1, (1, 1)), nn.Sigmoid())
                
        self.eps = 1e-8

    def forward(self, x):
        # x = (B, M, T)
        _x = x[:, 0]  # (B, T)

        batch_size = x.shape[0]
        n_mic = x.shape[1]
        
        win = torch.hann_window(self.FFT)
        
        if _x.is_cuda:
            win = win.cuda()
            
        # input shape: B, T
        all_mics = x.view(batch_size * n_mic, -1)  # (BxM, T)
        all_mics_stft = torch.stft(all_mics, self.FFT, self.HOP, window=win)  # (BxM, F, L, 2)

        all_mics_stft = all_mics_stft.contiguous().view(batch_size, n_mic, all_mics_stft.shape[1], all_mics_stft.shape[2], 2)  # (B, M, F, L, 2)
        all_mics_stft = all_mics_stft.permute(0, 1, 4, 2, 3)
        all_mics_stft = all_mics_stft.contiguous().view(-1, all_mics_stft.shape[2],
                                                        all_mics_stft.shape[3],
                                                        all_mics_stft.shape[4])  # (BxM, 2, F, L)

        feat = self.BN(self.enc_LN(all_mics_stft))  # (BxM, BN, F, L)

        # features est
        this_input1 = feat
        skip_connection1 = 0.
        for i in range(len(self.conv2)):
            this_output1 = self.conv2[i](this_input1)
            skip_connection1 = skip_connection1 + this_output1
            this_input1 = this_input1 + this_output1

        to_sum = skip_connection1.contiguous().view(batch_size, n_mic, skip_connection1.shape[1],
                                                    skip_connection1.shape[2],
                                                    skip_connection1.shape[3])  # (B, M, BN, F, L)

        summed = to_sum.sum(1)  # (B, BN, F, L)

        # mask estimation
        this_input = summed
        skip_connection = 0.
        for i in range(len(self.conv2)):
            this_output = self.conv2[i](this_input)
            skip_connection = skip_connection + this_output
            this_input = this_input + this_output

        mask_noise = self.reshape_noise(skip_connection).permute(0, 2, 1, 3).unsqueeze(2)  # B, F, 1, 1, T
        mask_speech = self.reshape_speech(skip_connection).permute(0, 2, 1, 3).unsqueeze(2)  # B, F, 1, 1, T
        # mask_noise = 1. - mask_speech

        observation = all_mics_stft.contiguous().view(batch_size, n_mic, 2,
                                                      all_mics_stft.shape[2],
                                                      all_mics_stft.shape[3]).permute(0, 3, 2, 1, 4)  # (B, F, 2, M, L)
        # calculate psds
        psd_noise = complex_psd(observation, mask_noise, normalize=False)  # (B, F, 2, M, M)
        psd_speech = complex_psd(observation, mask_speech, condition=True)  # (B, F, 2, M, M)

        psd_noise_cond = condition_covariance(psd_noise, 1e-6)
        psd_noise_norm = psd_noise_cond / trace(psd_noise_cond, dim1=-1, dim2=-2, keepdim=True)[:, :, 0].unsqueeze(2)

        # calculate weights
        # speech A
        # noise B
        re_a = psd_speech[:, :, 0]
        im_a = psd_speech[:, :, 1]
        re_b = psd_noise_norm[:, :, 0]
        im_b = psd_noise_norm[:, :, 1]

        a = torch.cat([torch.cat([re_a, -im_a], -1), torch.cat([im_a, re_a], -1)], -2)
        b = torch.cat([torch.cat([re_b, -im_b], -1), torch.cat([im_b, re_b], -1)], -2)
        h, _ = torch.gesv(a, b)  # (B, F, 2, M, M)

        trace_h = trace(h, keepdim=True)  # (B, F, 2, 1, 1)
        h_norm = h / trace_h

        h_re, h_im = h_norm[..., :n_mic, 0], h_norm[..., n_mic:, 0]

        # apply weights
        a = h_re  # (B, F, M)
        b = -h_im
        c = observation[:, :, 0].permute(0, 2, 1, 3)  # (B, M, F, L)
        d = observation[:, :, 1].permute(0, 2, 1, 3)  # (B, M, F, L)

        real = einsum(a, c, '...ab,...bac->...ac') - einsum(b, d, '...ab,...bac->...ac')
        imag = einsum(a, d, '...ab,...bac->...ac') + einsum(b, c, '...ab,...bac->...ac')  # (B, F, L)

        filtered = torch.cat([real.unsqueeze(-1), imag.unsqueeze(-1)], -1)  # B, F, T, 2
        output = my_istft(filtered, hop_length=self.HOP, cuda=filtered.is_cuda)  # B, T

        # filtered = mask_speech.squeeze(2) * x_stft.permute(0, 2, 1, 3)
        # output = my_istft(filtered.permute(0, 1, 3, 2), hop_length=self.HOP, cuda=filtered.is_cuda)  # B, T

        return output


class MudNoFFT(nn.Module):
    def __init__(self, n_fft=256, hop=125, learn_comp=False, bn_ch=32, sep_ch=64, kernel=(3, 3), causal=False, layers=6,
                 stacks=2, verbose=True):
        super(MudNoFFT, self).__init__()
        if verbose:
            print("NFFT IS: {}".format(n_fft))
        self.n_fft = (n_fft // 2 + 1)
        self.FFT = n_fft
        self.HOP = hop
        self.BN_channel = bn_ch
        self.conv_channel = sep_ch
        self.kernel = kernel
        self.conv_pad = (int(np.log2((self.kernel[0] - 1) / 2)), int(np.log2((self.kernel[1] - 1) / 2)))

        self.enc_LN = tLN(2)
        self.BN = nn.Conv2d(2, self.BN_channel, (1, 1))

        self.compression = nn.Parameter(torch.ones(1, ) * -0.8475, requires_grad=learn_comp)  # sigmoid(-0.8475) = 0.3
        if verbose:
            print("Compression: {}".format(torch.sigmoid(self.compression)))
        self.layer = layers
        self.stack = stacks

        self.receptive_field_time = 0
        self.receptive_field_freq = 0
        self.conv = nn.ModuleList([])
        for s in range(self.stack):
            for i in range(self.layer):
                self.conv.append(DepthConv2d(self.BN_channel, self.conv_channel,
                                             self.kernel, dilation=(2 ** i, 2 ** i), causal=causal,
                                             padding=(2 ** (i + self.conv_pad[0]), 2 ** (i + self.conv_pad[1]))))
                if s == 0 and i == 0:
                    self.receptive_field_time += self.kernel[0]
                    self.receptive_field_freq += self.kernel[1]
                else:
                    self.receptive_field_time += (self.kernel[0] - 1) * 2 ** i
                    self.receptive_field_freq += (self.kernel[1] - 1) * 2 ** i

        if verbose:
            print('Receptive field TIME: {:1d} samples.'.format(self.receptive_field_time))
            print('Receptive field FREQ: {:1d} samples.'.format(self.receptive_field_freq))

        self.reshape_env = nn.Sequential(nn.Conv1d(1, self.n_fft, 1),
                                         # nn.ReLU()
                                         )
        if verbose:
            print(self.reshape_env)

        self.reshape_speech = nn.Sequential(nn.Conv2d(self.BN_channel, 1, (1, 1)), nn.Sigmoid())

        self.eps = 1e-8

    def forward(self, x):
        # x = (B, 2, F, L)
        
        x_stft = x

        feat = self.BN(self.enc_LN(x_stft))  # (B, BN, F, L)
        # mask estimation
        print("Mask estimation")
        this_input = feat
        skip_connection = 0.
        for i in range(len(self.conv)):
            print("CONV {}".format(i))
            this_output = self.conv[i](this_input)
            skip_connection = skip_connection + this_output
            this_input = this_input + this_output
        
        print("Reshape")
        mask_speech = self.reshape_speech(skip_connection).permute(0, 2, 1, 3).unsqueeze(2)  # B, F, 1, 1, T

        return mask_speech


def compress(x, compression=0.3):
    # x: b, f, t, 2
    mag = torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2 + 1e-15)
    ang = torch.atan2(x[..., 1], x[..., 0])
    re = (mag ** compression) * torch.cos(ang)
    im = (mag ** compression) * torch.sin(ang)
    return torch.cat([re.unsqueeze(3), im.unsqueeze(3)], 3)


class Mud(nn.Module):
    def __init__(self, n_fft=256, hop=125, learn_comp=False, bn_ch=32, sep_ch=64, kernel=(3, 3), causal=False, layers=6,
                 stacks=2, verbose=True):
        super(Mud, self).__init__()
        if verbose:
            print("NFFT IS: {}".format(n_fft))
        self.n_fft = (n_fft // 2 + 1)
        self.FFT = n_fft
        self.HOP = hop
        self.BN_channel = bn_ch
        self.conv_channel = sep_ch
        self.kernel = kernel
        self.conv_pad = (int(np.log2((self.kernel[0] - 1) / 2)), int(np.log2((self.kernel[1] - 1) / 2)))

        self.enc_LN = tLN(2)
        self.BN = nn.Conv2d(2, self.BN_channel, (1, 1))

        self.compression = nn.Parameter(torch.ones(1, ) * -0.8475, requires_grad=learn_comp)  # sigmoid(-0.8475) = 0.3
        if verbose:
            print("Compression: {}".format(torch.sigmoid(self.compression)))
        self.layer = layers
        self.stack = stacks

        self.receptive_field_time = 0
        self.receptive_field_freq = 0
        self.conv = nn.ModuleList([])
        for s in range(self.stack):
            for i in range(self.layer):
                self.conv.append(DepthConv2d(self.BN_channel, self.conv_channel,
                                             self.kernel, dilation=(2 ** i, 2 ** i), causal=causal,
                                             padding=(2 ** (i + self.conv_pad[0]), 2 ** (i + self.conv_pad[1]))))
                if s == 0 and i == 0:
                    self.receptive_field_time += self.kernel[0]
                    self.receptive_field_freq += self.kernel[1]
                else:
                    self.receptive_field_time += (self.kernel[0] - 1) * 2 ** i
                    self.receptive_field_freq += (self.kernel[1] - 1) * 2 ** i

        if verbose:
            print('Receptive field TIME: {:1d} samples.'.format(self.receptive_field_time))
            print('Receptive field FREQ: {:1d} samples.'.format(self.receptive_field_freq))

        self.reshape_env = nn.Sequential(nn.Conv1d(1, self.n_fft, 1),
                                         # nn.ReLU()
                                         )
        if verbose:
            print(self.reshape_env)

        # self.reshape_noise = nn.Sequential(nn.Conv2d(self.BN_channel, 1, (1, 1)), nn.Sigmoid())
        self.reshape_speech = nn.Sequential(nn.Conv2d(self.BN_channel, 1, (1, 1)), nn.Sigmoid())

        self.eps = 1e-8

    def forward(self, x):
        # x = (B, M, T)
        _x = x[:, 0]  # (B, T)

        batch_size = x.shape[0]
        n_mic = x.shape[1]

        win = torch.hann_window(self.FFT)

        if _x.is_cuda:
            win = win.cuda()

        # input shape: B, T
        all_mics = x.view(batch_size * n_mic, -1)  # (BxM, T)
        all_mics_stft = torch.stft(all_mics, self.FFT, self.HOP, window=win)  # (BxM, F, L, 2)

        all_mics_stft = all_mics_stft.contiguous().view(batch_size, n_mic, all_mics_stft.shape[1],
                                                        all_mics_stft.shape[2], 2)  # (B, M, F, L, 2)
        x_stft = all_mics_stft[:, 0]  # (B, F, L, 2)

        x_stft = x_stft.permute(0, 3, 1, 2)  # (B, 2, F, L)

        feat = self.BN(self.enc_LN(x_stft))  # (B, BN, F, L)
        # mask estimation

        this_input = feat
        skip_connection = 0.
        for i in range(len(self.conv)):
            this_output = self.conv[i](this_input)
            skip_connection = skip_connection + this_output
            this_input = this_input + this_output

        # mask_speech = self.reshape_speech(skip_connection).permute(0, 2, 1, 3).unsqueeze(2)  # B, F, 1, 1, T
        mask_speech = self.reshape_speech(skip_connection).permute(0, 2, 1, 3).unsqueeze(2)  # B, F, 1, 1, T
        mask_noise = 1. - mask_speech

        observation = all_mics_stft.permute(0, 2, 4, 1, 3)  # (B, F, 2, M, L)
        # calculate psds
        psd_noise = complex_psd(observation, mask_noise, normalize=False)  # (B, F, 2, M, M)
        psd_speech = complex_psd(observation, mask_speech, condition=True)  # (B, F, 2, M, M)

        psd_noise_cond = condition_covariance(psd_noise, 1e-6)
        psd_noise_norm = psd_noise_cond / trace(psd_noise_cond, dim1=-1, dim2=-2, keepdim=True)[:, :, 0].unsqueeze(2)

        # calculate weights
        # speech A
        # noise B
        re_a = psd_speech[:, :, 0]
        im_a = psd_speech[:, :, 1]
        re_b = psd_noise_norm[:, :, 0]
        im_b = psd_noise_norm[:, :, 1]

        a = torch.cat([torch.cat([re_a, -im_a], -1), torch.cat([im_a, re_a], -1)], -2)
        b = torch.cat([torch.cat([re_b, -im_b], -1), torch.cat([im_b, re_b], -1)], -2)
        h, _ = torch.gesv(a, b)  # (B, F, 2, M, M)

        trace_h = trace(h, keepdim=True)  # (B, F, 2, 1, 1)
        h_norm = h / trace_h

        h_re, h_im = h_norm[..., :n_mic, 0], h_norm[..., n_mic:, 0]

        # apply weights
        a = h_re  # (B, F, M)
        b = -h_im
        c = observation[:, :, 0].permute(0, 2, 1, 3)  # (B, M, F, L)
        d = observation[:, :, 1].permute(0, 2, 1, 3)  # (B, M, F, L)

        real = einsum(a, c, '...ab,...bac->...ac') - einsum(b, d, '...ab,...bac->...ac')
        imag = einsum(a, d, '...ab,...bac->...ac') + einsum(b, c, '...ab,...bac->...ac')  # (B, F, L)

        filtered = torch.cat([real.unsqueeze(-1), imag.unsqueeze(-1)], -1)  # B, F, T, 2
        output = my_istft(filtered, hop_length=self.HOP, cuda=filtered.is_cuda)  # B, T

        # filtered = mask_speech.squeeze(2) * x_stft.permute(0, 2, 1, 3)
        # output = my_istft(filtered.permute(0, 1, 3, 2), hop_length=self.HOP, cuda=filtered.is_cuda)  # B, T

        return output