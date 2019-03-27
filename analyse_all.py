import warnings

import numpy as np
from scipy.signal import resample_poly
import itertools
from collections import OrderedDict

import csv
import soundfile as sf

from beamformers import beamformers as bf
from sep_eval import sep_eval as se
from librosa import stft


warnings.simplefilter(action='ignore', category=FutureWarning)

NOISE_CALIB = np.load('noise_calib.npy')
FS = 24000
SESSION = 2


# trigger (kinda)
def find_trigger_v2(x):
    Y = x
    X = stft(Y)
    Y = np.log10(np.abs(X) ** 2 + 1e-8)
    YY = np.sum(Y, 0)
    n90 = np.percentile(YY, 65)
    tr = np.where(YY >= n90)[0][0]
    return tr * 512


def single_output(rec, reference, interference, noise_calib, ch, mode='sep'):
    # assumes full rec and calib
    all_res = OrderedDict()
    all_err = OrderedDict()
    for method, name in zip([bf.SDW_MWF], ['sdw_mwf']):
        all_err[name] = []
        _sep_sources = []
        _ori_sources = []
        _single_eval = []
        print("\t\t\tDoing method: %s" % name)

        _full_set = set(range(len(sources)))

        interference = noise_calib
        record = rec[:, ch].T
        noise = noise_calib[:, ch].T

        denoised_gt_1 = method(reference, interference=noise, reference=None, frame_len=1024, frame_step=128)

        recon = method(record, interference=interference, reference=reference, frame_len=1024, frame_step=128)

        # test quality on first 15 seconds
        y_hat = recon[:10 * FS]
        y = denoised_gt_1

        # calculate best correlation lag
        corr = np.correlate(y_hat - np.mean(y_hat), y - np.mean(y), mode='full')
        lag = corr.argmax() - (len(y_hat) - 1)
        _err_log = "OK"

        if lag > 0:
            y_hat_2 = y_hat[lag:]
            y_2 = y[:-lag]
        else:
            y_hat_2 = y_hat[:]
            y_2 = y[:]
            _err_log = "ERROR: lag {}".format(lag)

        all_err[name].append(_err_log)

        new_fs = 16000
        y_2_16k = resample_poly(y_2, 2, 3)
        y_hat_2_16k = resample_poly(y_hat_2, 2, 3)

        _sep_sources.append(y_hat_2_16k)
        _ori_sources.append(y_2_16k)

        _single_eval.append(se.full_eval(y_hat_2_16k, y_2_16k, fs=new_fs))

    return all_res, all_err


def main():
    # collect files
    timestamps = []
    speakers = []
    valids = []
    all_nspk = OrderedDict()

    # mapping modules to channels
    channel_map = {5: np.array([0, 1, 2, 3]),  # random
                   6: np.array([4, 5, 6, 7]),  # line
                   7: np.array([8, 9, 10, 11]),  # round
                   8: np.array([12, 13, 14, 15])}  # random

    with open("output_session2_15s.csv", "r") as f:
        reader = csv.reader(f, delimiter=",")
        for i, line in enumerate(reader):
            all_nspk[line[0]] = int(line[-1])
            timestamps.append(line[0])
            speakers.append(np.array(line[1:5]))
            _traces = line[5:-1]
            valids.append(np.array(np.where(np.array(_traces) != '/Users/enea/DATASETS/Noise/silence.wav')[0]))

    print("Collected %d files" % len(timestamps))

    with open('debug_sess{}_analysis_all_v5.csv'.format(SESSION), 'w') as csv_file:

        file_writer = csv.writer(csv_file, delimiter=',')
        header = ['name', 'speakers', 'ts5', 'ts6', 'ts7', 'ts8', 'exp_time_true', 'exp_time_emp', 'nspk', 'ch_subset', 'method', 'active',
                  'bss_sar', 'bss_sdr', 'bss_sir', 'e_stoi', 'pesq', 'sdr', 'stoi', 'errors']
        file_writer.writerow(header)

        for timestamp, speaker, valid in zip(timestamps, speakers, valids):  # timestamp
            full_row = []
            print("Processing: %s" % timestamp)
            full_row.append(timestamp)
            print("\t{} Sources in {}".format(len(valid), speaker[valid]))
            full_row.append("-".join(speaker[valid]))

            all_data = []
            _t_secs = []
            # load 4 traces from whispers
            for i in [5, 6, 7, 8]:
                _y, _FS = sf.read('../data_whisper/session%d/test_%s_wh%d.wav' % (SESSION, timestamp, i))
                all_data.append(_y)
                assert (_FS == FS)

                t_secs = _y.shape[0] // FS
                _t_secs.append(t_secs)
                full_row.append(t_secs)

            # check that the platforms are sync
            avail_channels = channel_map[5]
            avail_modules = [5]

            for jj, p in zip(range(1, 4), [6, 7, 8]):
                if abs(_t_secs[0] - _t_secs[jj]) <= 2:
                    avail_channels = np.append(avail_channels, channel_map[p])
                    avail_modules.append(p)

            print("\tAvailable channels {}".format(avail_channels))
            print("\tAvailable modules {}".format(avail_modules))

            n_spk = all_nspk[timestamp]

            # cut to shortest
            m_len = np.min([a.shape[0] for a in all_data])
            all_data_cut = [a[:m_len] for a in all_data]
            all_data_array = np.hstack(all_data_cut).astype('float32')

            # find beginning (assumes silence before)
            trigger = find_trigger_v2(all_data_array[:, 4])

            # extract 4 calib phases and actual recording
            noise_calib = all_data_array[:trigger]
            calib = [all_data_array[i * 15 * FS + trigger: ((i*15) + 10) * FS + trigger] for i in range(max(2, n_spk))]
            if SESSION == 2:
                rec = all_data_array[max((15 * n_spk), 30) * FS + trigger:]
                full_row.append(max((15 * n_spk), 30) + 15)
                full_row.append(_t_secs[0] - trigger // FS)
            elif SESSION == 1:
                rec = all_data_array[60 * FS + trigger:]
                full_row.append(180)
                full_row.append(_t_secs[0] - trigger // FS)
            else:
                raise ValueError("WRONG SESSION")

            full_row.append(n_spk)

            if any([c.shape[0] != 10 * FS for c in calib]) or rec.shape[0] < 10 * FS:
                error = ['N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
                         'ERROR: Probable trigger problem']
                file_writer.writerow(full_row + error)
                continue

            # create combinations of channels
            # IMPORTANT consider module only if it LOOKS like it is sync
            # 2 for each of 2, 4, 6, 8, 10, 12, 14, 16  (or max if not using 16 mics)
            # single platforms (if available)

            all_comb = []
            for p in avail_modules:
                all_comb.append(channel_map[p])  # available single platforms

            for n_ch in range(2, len(avail_channels) + 1, 2):
                for i in range(2):
                    all_comb.append(np.sort(np.random.choice(avail_channels, n_ch, replace=False)))

            for ch_subset in all_comb:  # combination
                comb_row = []
                print("\t\tCombination: {}".format(ch_subset))
                comb_row.append("-".join([str(c) for c in ch_subset]))

                _res, _errors = single_output(rec, [calib[i] for i in range(max(n_spk, 2))], noise_calib, ch_subset,
                                              mode='sep' if n_spk >= 2 else'enh')  #

                for method in sorted(_res.keys()):  # method
                    method_row = [method]
                    for jj, _r in enumerate(_res[method]):  # speaker
                        speaker_row = [speaker[jj]]
                        _e = _errors[method][jj]
                        for _k in sorted(_r.keys()):  # measure
                            speaker_row.append(float(_r[_k]))
                        row_to_write = list(itertools.chain.from_iterable([full_row, comb_row, method_row,
                                                                           speaker_row, [_e]]))
                        file_writer.writerow(row_to_write)


if __name__ == "__main__":
    main()
