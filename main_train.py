import argparse
import math
import random
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import StegNet
from utils import lora_dataset, WmkDataset, LRScheduler, EarlyStopping, wgn_torch


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--keylen', type=int, default=180, help='key size')
    parser.add_argument('--B', type=int, default=64, help='local batch size')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    parser.add_argument('--fs', type=float, default=1e6, help='sampling frequency')
    parser.add_argument('--sf', type=int, default=7, help='spreading factor')
    parser.add_argument('--bw', type=float, default=125e3, help='bandwidth')

    args = parser.parse_args()

    return args


class Train:
    def __init__(self, args):

        self.args = args

    def _cfo(self, tensor):

        offset = random.uniform(-5, 5)  # around 25 ppm
        t = torch.arange(1, len(tensor) + 1) * 1 / args.fs
        tensor = tensor * torch.exp(1j * 2 * torch.tensor(math.pi).to(tensor.device) * offset * t.to(tensor.device))

        return tensor

    # def _awgn(self, tensor, mean=0, stddev=1):  
    #     noise = torch.nn.init.normal_(torch.Tensor(tensor.size()), mean, stddev) + 1j*torch.nn.init.normal_(torch.Tensor(tensor.size()), mean, stddev)
    #     return tensor + noise.to(tensor.device)

    # def _awgn_(self, tensor, SNRdB):

    #     SNR_linear = 10 ** (SNRdB / 10)
    #     N0 = torch.sum(torch.abs(tensor) ** 2) / (torch.len(tensor)*SNR_linear)
    #     # N0 = P / SNR_linear
    #     # n = np.sqrt(N0 / 2) * (np.random.standard_normal(len(s)) + 1j * np.random.standard_normal(len(s)))
    #     noise = torch.sqrt(N0 / 2) * (torch.nn.init.normal_(torch.Tensor(tensor.size())) + 1j*torch.nn.init.normal_(torch.Tensor(tensor.size()))).to(tensor.device)

    #      # noise = torch.nn.init.normal_(torch.Tensor(tensor.size()), mean=0, stddev=1) + 1j*torch.nn.init.normal_(torch.Tensor(tensor.size()), mean=0, stddev=1)
    #     return tensor + noise

    def start_train(self):

        dummy_dataset = lora_dataset(sf=args.sf, bw=args.bw, fs=args.fs, num_symbols=1,
                                     num_packets=5000)  # 5000 previously

        data_train, data_valid = train_test_split(dummy_dataset,
                                                  test_size=0.1,
                                                  shuffle=True)
        print('Loading done.')

        dataset_train = WmkDataset(data_train, keylen=args.keylen)
        dataset_valid = WmkDataset(data_valid, keylen=args.keylen)

        trainloader = DataLoader(dataset_train,
                                 batch_size=args.B,
                                 shuffle=True,
                                 num_workers=0,
                                 drop_last=True)

        validloader = DataLoader(dataset_valid,
                                 batch_size=args.B,
                                 shuffle=True,
                                 num_workers=0,
                                 drop_last=True)

        model = StegNet(key_len=args.keylen).to(args.device)

        for module in model.modules():
            if isinstance(module, torch.nn.Conv1d):
                # torch.nn.init.xavier_normal_(module.weight)
                # torch.nn.init.normal_(module.weight)
                torch.nn.init.kaiming_normal_(module.weight)

        print("The model will be running on", args.device, "device")

        # Training parameters
        epochs = 2000

        optimizer = torch.optim.RMSprop(model.parameters(), lr=3e-4)
        # optimizer =  torch.optim.Adam(model.parameters(), lr = 3e-4)

        lr_scheduler = LRScheduler(optimizer)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
        # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        early_stopping = EarlyStopping()

        training_loss, training_acc = [], []
        valid_loss, valid_acc = [], []

        cos_sim_loss = torch.nn.CosineEmbeddingLoss()
        bce_loss = torch.nn.BCELoss()
        mse_loss = torch.nn.MSELoss()

        print('Start training.')
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

        alpha = 0.6  # weight for the key loss
        for epoch in range(epochs):  # loop over the dataset multiple times

            model.train()
            training_running_loss = 0.0
            training_running_ber = 0.0

            for iteration, (covers, secrets) in enumerate(tqdm(trainloader)):

                covers = covers.to(args.device)
                secrets = secrets.to(args.device)

                optimizer.zero_grad()

                # containers = model.encode(secrets, covers)

                containers = model.encode(secrets, covers)

                # containers = torch.complex(containers[:, 0, :], containers[:, 1, :])

                loss_wave = mse_loss(torch.cat([covers.real, covers.imag], dim=1),
                                     torch.cat([containers.real, containers.imag], dim=1))

                for i in range(args.B):
                    # containers[i] = self._cfo(containers[i])
                    # ch_coef = rayleigh_coef_torch(args)
                    theta = np.random.uniform(low=-np.pi, high=np.pi)
                    # containers[i] = containers[i] * ch_coef
                    # containers[i] = containers[i] + wgn_torch(args, (covers[i]*ch_coef).squeeze(), snr_min = 0, snr_max = 60)
                    containers[i] = containers[i] * np.exp(1j * theta) + wgn_torch(args, containers[i], snr_min=0,
                                                                                   snr_max=60)
                    # containers[i] = containers[i] * ch_coef.item()
                    # containers[i] = self._awgn(containers[i], mean=0, stddev=random.uniform(0, 0.3))

                # containers_noisy = self._awgn(containers, mean=0, stddev=random.uniform(0, 1))

                secrets_restored = model.decode(containers)

                loss_key = mse_loss(secrets, secrets_restored)

                # loss_wave = mse_loss(torch.cat([containers_fft.real, containers_fft.imag], dim=1), torch.cat([covers_fft.real, covers_fft.imag], dim=1))

                # loss_wave = cos_sim_loss(torch.cat([covers.real, covers.imag], dim=1), torch.cat([containers.real, containers.imag], dim=1), target=torch.ones(args.B).to(args.device))

                loss = alpha * loss_key + (1 - alpha) * loss_wave

                loss.backward()
                optimizer.step()

                # print statistics
                training_running_loss += loss.item()
                training_running_ber += (secrets != torch.sign(secrets_restored)).sum().item() / secrets.numel()

            training_epoch_loss = training_running_loss / (iteration + 1)
            training_epoch_ber = training_running_ber / (iteration + 1)

            model.eval()
            valid_running_loss = 0.0
            valid_running_ber = 0.0

            for iteration, (covers, secrets) in enumerate(validloader):

                covers = covers.to(args.device)
                secrets = secrets.to(args.device)

                containers_v = model.encode(secrets, covers)

                loss_wave_v = mse_loss(torch.cat([covers.real, covers.imag], dim=1),
                                       torch.cat([containers_v.real, containers_v.imag], dim=1))
                # containers_v = torch.complex(containers_v[:, 0, :], containers_v[:, 1, :])

                # if epoch > 10:
                for i in range(args.B):
                    # containers_v[i] = self._cfo(containers_v[i])
                    # ch_coef = rayleigh_coef_torch(args)
                    theta = np.random.uniform(low=-np.pi, high=np.pi)

                    # containers_v[i] = containers_v[i] * ch_coef
                    # containers_v[i] = containers_v[i] + wgn_torch(args, (covers[i]*ch_coef).squeeze(), snr_min = 0, snr_max = 60)

                    containers_v[i] = containers_v[i] * np.exp(1j * theta) + wgn_torch(args, containers_v[i], snr_min=0,
                                                                                       snr_max=60)

                    # containers_v[i] = containers_v[i] * ch_coef.item()
                    # containers_v[i] = self._awgn(containers_v[i], mean=0, stddev=random.uniform(0, 0.3))

                secrets_restored_v = model.decode(containers_v)

                loss_key_v = mse_loss(secrets, secrets_restored_v)
                # loss_wave_v = mse_loss(containers_fft_amp_v, covers_fft_amp_v)
                # loss_wave_v = mse_loss(torch.cat([containers_fft_v.real, containers_fft_v.imag], dim=1), torch.cat([covers_fft_v.real, covers_fft_v.imag], dim=1))

                # loss_wave_v = cos_sim_loss(torch.cat([covers.real, covers.imag], dim=1), torch.cat([containers_v.real, containers_v.imag], dim=1), target=torch.ones(args.B).to(args.device))

                # loss_wave_v = cos_sim_loss(wave_wmk_v, inputs, torch.ones(args.B).to(args.device))

                loss = alpha * loss_key_v + (1 - alpha) * loss_wave_v

                # print statistics
                valid_running_loss += loss.item()
                valid_running_ber += (secrets != torch.sign(secrets_restored_v)).sum().item() / secrets.numel()

            valid_epoch_loss = valid_running_loss / (iteration + 1)
            valid_epoch_ber = valid_running_ber / (iteration + 1)

            # print('Epoch ' + str(epoch + 1) + '\t Training Loss: ' + str(
            #     training_epoch_loss) + '\t Validation Loss: ' + str(valid_epoch_loss))

            print('Epoch ' + str(epoch + 1)
                  + '\t Training Loss: ' + str(round(training_epoch_loss, 3))
                  + '\t Validation Loss: ' + str(round(valid_epoch_loss, 3))
                  + '\t Training BER: ' + str(round(training_epoch_ber, 3))
                  + '\t Validation BER: ' + str(round(valid_epoch_ber, 3))
                  # +'\t LR: '+str(round(lr_scheduler.get_last_lr()[0], 4))
                  + '\t')

            training_loss.append(training_epoch_loss)
            training_acc.append(training_epoch_ber)
            valid_loss.append(valid_epoch_loss)
            valid_acc.append(valid_epoch_ber)

            # best_model = copy.deepcopy(model)
            # best_clf = copy.deepcopy(metric_fc)

            # lr_scheduler.step()
            lr_scheduler(valid_epoch_loss)
            early_stopping(valid_epoch_loss)
            if early_stopping.early_stop:
                break

        print('Finished Training')
        print('Training done.')
        return model


if __name__ == '__main__':
    args = args_parser()
    train = Train(args)

    model = train.start_train()
    torch.save(model.state_dict(), './model/nn_keylen_' + str(args.keylen) + '_bw_' + str(int(args.bw / 1e3)) + '.pth')
