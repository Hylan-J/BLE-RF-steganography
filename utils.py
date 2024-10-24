#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description: 辅助类以及相关函数
import numpy as np
from torch.utils.data import Dataset
import torch


def rayleigh_coef_torch(args, omega=1):
    h_rayleigh = torch.sqrt(torch.tensor(omega / 2)) * (torch.complex(torch.randn(1, 1), torch.randn(1, 1)))
    return h_rayleigh.to(args.device)


def split_complex(x, method='real_imag'):
    x = x.unsqueeze(1)

    if method == 'real_imag':
        x_real = x.real
        x_imag = x.imag
        out = torch.cat([x_real, x_imag], dim=1)
    elif method == 'mag_phase':
        x_mag = torch.abs(x)
        x_phase = torch.angle(x)
        out = torch.cat([x_mag, x_phase], dim=1)
    else:
        print('ERROR: Invalid spliting method')

    return out


def lora_symbol(sf, bw, fs, num_symbols):
    # Ts = 1/fs
    N = 2 ** sf
    T = N / bw
    # samp_per_sym = round(fs/bw*N)
    samp_per_sym = round((2 ** sf / bw) * fs)
    k = bw / T

    f0 = -bw / 2
    t = np.arange(0, samp_per_sym) * (1 / fs)
    base_chirp = np.exp(1j * 2 * np.pi * (t * (f0 + 0.5 * k * t)))

    t = np.arange(0, num_symbols * samp_per_sym) * (1 / fs)
    sig = np.tile(base_chirp, (1, num_symbols))
    return t, sig


def lora_dataset(sf=7, bw=125000, num_symbols=8, fs=125000, num_packets=500):
    t, sig = lora_symbol(sf, bw, fs, num_symbols)
    dummy_dataset = np.tile(sig, (num_packets, 1))
    return dummy_dataset

#dummy_dataset = lora_dataset(sf=7, bw=125e3, fs=1e6, num_symbols=1, num_packets=5000)
#print(dummy_dataset.shape)



def cfo_compensation(data, cfo, Ts):
    n = np.arange(1, len(data) + 1)
    data_shift = data * np.exp(-1j * 2 * np.pi * cfo * n * Ts)
    return data_shift


def cal_ber(x, y, verbose=False):
    index = np.arange(x.size)
    pos = index[x != y]
    hamming_dist = np.sum(x != y)
    ber = hamming_dist / x.size
    if verbose:
        print('Bits Number: ', x.size, 'Hamming Dist:', hamming_dist, ', BER: ', ber, ', Error Position:', pos)

    return ber, hamming_dist


def cal_exponential_pdp(tau_d, Ts, A_dB=-30):
    # Exponential PDP generator
    # Inputs:
    # tau_d : rms delay spread[sec]
    # Ts : Sampling time[sec]
    # A_dB : smallest noticeable power[dB]
    # norm_flag : normalize total power to unit
    # Output:
    # PDP : PDP vector

    sigma_tau = tau_d
    A = 10 ** (A_dB / 10)
    lmax = np.ceil(-tau_d * np.log(A) / Ts)

    # Exponential PDP
    p = np.arange(0, lmax + 1)
    pathDelays = p * Ts

    p = (1 / sigma_tau) * np.exp(-p * Ts / sigma_tau)
    p_norm = p / np.sum(p)

    avgPathGains = 10 * np.log10(p_norm)

    return avgPathGains, pathDelays


def wgn_torch(args, sig_in, snr_min, snr_max):
    SNR_dB = np.random.uniform(snr_min, snr_max)
    SNR_linear = 10 ** (SNR_dB / 10)

    P = sum(abs(sig_in.detach().cpu().numpy()) ** 2) / len(sig_in.detach().cpu().numpy())
    N0 = P / SNR_linear

    noise = np.sqrt(N0 / 2) * (torch.complex(torch.randn(sig_in.size()), torch.randn(sig_in.size())))

    return noise.to(args.device)


def awgn(data, snr_range):
    if len(data.shape) == 1:
        data = data.reshape(1, len(data))

    data_noisy = np.zeros(data.shape, dtype=complex)
    pkt_num = data.shape[0]
    SNRdB = np.random.uniform(snr_range[0], snr_range[-1], pkt_num)
    for pktIdx in range(pkt_num):
        s = data[pktIdx]
        # SNRdB = uniform(snr_range[0],snr_range[-1])
        SNR_linear = 10 ** (SNRdB[pktIdx] / 10)
        P = sum(abs(s) ** 2) / len(s)
        N0 = P / SNR_linear
        n = np.sqrt(N0 / 2) * (np.random.standard_normal(len(s)) + 1j * np.random.standard_normal(len(s)))
        data_noisy[pktIdx] = s + n

    return data_noisy


# ----------------------------------------------------------------------------------------------------------------------
# 相关类的定义
# ----------------------------------------------------------------------------------------------------------------------
class WmkDataset(Dataset):
    def __init__(self, data_input, keylen):
        self.data = data_input
        self.keylen = keylen

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_input = self.data[index]
        keys = np.random.choice([-1, 1], size=self.keylen)

        data_input = torch.from_numpy(data_input.astype(np.complex64))
        keys = torch.from_numpy(keys.astype(np.float32))
        return data_input, keys


class LRScheduler:
    def __init__(self, optimizer, patience=10, min_lr=1e-6, factor=0.1):
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.patience,
            factor=self.factor,
            min_lr=self.min_lr,
            verbose=True
        )

    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)


class EarlyStopping:
    def __init__(self, patience=20, min_delta=0):
        self.min_delta = min_delta
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            # print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
