import numpy as np
from utils import awgn, lora_dataset, WmkDataset, LRScheduler, EarlyStopping, split_complex, wgn_torch, \
    rayleigh_coef_torch, cal_ber
import matplotlib.pyplot as plt
import time
import argparse
import torch
from model import StegNet
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from scipy.io import savemat
import math
import random

from numpy.fft import fft, ifft, fftshift


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--keylen', type=int, default=180, help='key size')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    parser.add_argument('--fs', type=float, default=1e6, help='sampling frequency')
    parser.add_argument('--sf', type=int, default=7, help='spreading factor')
    parser.add_argument('--bw', type=float, default=125e3, help='bandwidth')

    args = parser.parse_args()

    return args


class Test:
    def __init__(self, args):
        self.args = args

    def gaussian_noise(self, tensor, mean=0, stddev=1):
        noise = torch.nn.init.normal_(torch.Tensor(tensor.size()), mean, stddev)
        return tensor + noise.to(tensor.device)

    def _cfo(self, tensor):

        offset = random.uniform(-10, 10)  # around 25 ppm
        t = torch.arange(1, len(tensor) + 1) * 1 / self.args.fs
        tensor = tensor * torch.exp(1j * 2 * torch.tensor(math.pi).to(tensor.device) * offset * t.to(tensor.device))

        return tensor

    def _awgn(self, tensor, mean=0, stddev=1):
        noise = torch.nn.init.normal_(torch.Tensor(tensor.size()), mean, stddev) + 1j * torch.nn.init.normal_(
            torch.Tensor(tensor.size()), mean, stddev)
        return tensor + noise.to(tensor.device)

    def start_test(self):
        data = lora_dataset(sf=args.sf, bw=args.bw, fs=args.fs, num_symbols=1, num_packets=1000)

        model = StegNet(key_len=args.keylen).to(args.device)
        model.load_state_dict(
            torch.load('./model/nn_keylen_' + str(args.keylen) + '_bw_' + str(int(args.bw / 1e3)) + '.pth'))
        model.eval()

        mse_loss = torch.nn.MSELoss()

        pkt_error = 0
        pkt_error_hamming = 0
        ber_all = []
        ber_hamming_all = []

        for data_ind in range(len(data)):
            std_chirp = data[data_ind]
            # std_chirp = std_chirp/np.sqrt(np.mean(np.abs(std_chirp)**2)) # normalize the preamble

            std_chirp = np.expand_dims(std_chirp, axis=0)
            std_chirp = torch.from_numpy(std_chirp.astype(np.complex64)).to(args.device)

            hamming_code = False

            if hamming_code:

                message_source = np.random.choice([0, 1], size=args.keylen)  # generate random 32 bits

                # message_hamming_encode = hamming.encode(message_source)
                # message_interleaved = interleaver.interleave(message_hamming_encode)

            else:
                message_source = np.random.choice([0, 1], size=args.keylen)  # generate random 56 bits
                message_interleaved = message_source.copy()

            ##########################
            # Transmitter operations #
            ##########################

            message_interleaved = np.expand_dims(message_interleaved, axis=0)
            message_interleaved = torch.from_numpy(message_interleaved.astype(np.float32)).to(args.device)
            message_interleaved[message_interleaved == 0] = -1

            container = model.encode(message_interleaved, std_chirp)

            ########################
            #  Receiver operations #
            ########################

            # container = self._cfo(container)
            # container = container * rayleigh_coef_torch(args)
            container = container + wgn_torch(args, container.squeeze(), snr_min=60, snr_max=60)
            # container = container

            message_nn_decoded = model.decode(container)  # container or container_noisy
            message_nn_decoded = torch.sign(message_nn_decoded)

            message_nn_decoded = message_nn_decoded.cpu().detach().numpy().astype(int).squeeze()
            message_nn_decoded[message_nn_decoded == -1] = 0

            message_interleaved = message_interleaved.cpu().detach().numpy().astype(int).squeeze()
            message_interleaved[message_interleaved == -1] = 0

            if hamming_code:

                # message_deinterleaved = interleaver.deinterleave(message_nn_decoded)

                # message_hamming_decoded = hamming.decode(message_deinterleaved)
                print('Decoded message:', message_nn_decoded)
            else:
                message_hamming_decoded = message_nn_decoded

            ber, ham_dist = cal_ber(message_interleaved, message_nn_decoded, verbose=True)

            ber_hamming, ham_dist_hamming = cal_ber(message_source, message_hamming_decoded, verbose=True)

            loss_wave = mse_loss(split_complex(std_chirp), split_complex(container))

            if ham_dist != 0:
                pkt_error += 1

            if ham_dist_hamming != 0:
                pkt_error_hamming += 1

            ber_all.append(ber)
            ber_hamming_all.append(ber_hamming)
            # print('Data index:', data_ind)
            plot = False
            if plot:
                container_tmp = container.cpu()[0, :].detach().numpy()
                cover_tmp = std_chirp[0].cpu().detach().numpy()

                plt.figure()
                plt.subplot(611)
                plt.plot(np.real(container_tmp), color='r')
                plt.subplot(612)
                plt.plot(np.abs(fftshift(fft(container_tmp))))
                # plt.subplot(613)
                # plt.plot(np.abs(container_tmp))

                plt.subplot(614)
                plt.plot(np.real(cover_tmp), color='b')
                plt.subplot(615)
                plt.plot(np.abs(fftshift(fft(cover_tmp))))
                # plt.subplot(616)
                # plt.plot(np.abs(cover_tmp))
                # plt.ylim(0, 2)
                plt.ticklabel_format(useOffset=False)

                # plt.legend(['Distorted signal', 'Original signal'])
                plt.show()

        print('Average BER:', np.mean(ber_all), ', Average Hamming BER:', np.mean(ber_hamming_all),
              ', Packet error rate:', pkt_error / len(data), ', Packet error rate Hamming:',
              pkt_error_hamming / len(data))


if __name__ == '__main__':
    args = args_parser()
    test = Test(args)
    test.start_test()
