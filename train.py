#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description: 比特嵌入器、比特提取器联合训练部分
import argparse
import time

import numpy as np
import torch
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import StegNet
from utils import WmkDataset, LRScheduler, EarlyStopping, wgn_torch


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--keylen', type=int, default=60, help='key size')
    parser.add_argument('--B', type=int, default=64, help='local batch size')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    args = parser.parse_args()

    return args


class Train:
    def __init__(self, args):
        self.args = args

    def start_train(self):
        f = loadmat('./matlab_code/preamble_waveform_5000.mat')
        dummy_dataset = f['preamble']
        print(dummy_dataset.shape)
        data_train, data_valid = train_test_split(dummy_dataset, test_size=0.1, shuffle=True)
        print('Loading done.')

        dataset_train = WmkDataset(data_train, keylen=args.keylen)
        dataset_valid = WmkDataset(data_valid, keylen=args.keylen)
        trainloader = DataLoader(dataset_train, batch_size=args.B, shuffle=True, drop_last=True)
        validloader = DataLoader(dataset_valid, batch_size=args.B, shuffle=True, drop_last=True)

        model = StegNet(key_len=args.keylen).to(args.device)
        for module in model.modules():
            if isinstance(module, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(module.weight)
        print("The model will be running on", args.device, "device")

        # Training parameters
        epochs = 2000

        optimizer = torch.optim.RMSprop(model.parameters(), lr=3e-4)
        lr_scheduler = LRScheduler(optimizer)
        early_stopping = EarlyStopping()
        mse_loss = torch.nn.MSELoss()

        training_loss, training_acc = [], []
        valid_loss, valid_acc = [], []

        print('Start training.')
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

        alpha = 0.6  # weight for the key loss
        for epoch in range(epochs):

            model.train()
            training_running_loss = 0.0
            training_running_ber = 0.0
            for iteration, (covers, secrets) in enumerate(tqdm(trainloader)):
                covers = covers.to(args.device)
                secrets = secrets.to(args.device)
                optimizer.zero_grad()
                containers = model.encode(secrets, covers)
                loss_wave = mse_loss(torch.cat([covers.real, covers.imag], dim=1),
                                     torch.cat([containers.real, containers.imag], dim=1))

                for i in range(args.B):
                    theta = np.random.uniform(low=-np.pi, high=np.pi)
                    containers[i] = containers[i] * np.exp(1j * theta) + wgn_torch(args, containers[i], snr_min=0,
                                                                                   snr_max=60)

                secrets_restored = model.decode(containers)
                loss_key = mse_loss(secrets, secrets_restored)
                loss = alpha * loss_key + (1 - alpha) * loss_wave

                loss.backward()
                optimizer.step()

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

                for i in range(args.B):
                    theta = np.random.uniform(low=-np.pi, high=np.pi)
                    containers_v[i] = containers_v[i] * np.exp(1j * theta) + wgn_torch(args, containers_v[i], snr_min=0,
                                                                                       snr_max=60)

                secrets_restored_v = model.decode(containers_v)
                loss_key_v = mse_loss(secrets, secrets_restored_v)
                loss = alpha * loss_key_v + (1 - alpha) * loss_wave_v

                valid_running_loss += loss.item()
                valid_running_ber += (secrets != torch.sign(secrets_restored_v)).sum().item() / secrets.numel()

            valid_epoch_loss = valid_running_loss / (iteration + 1)
            valid_epoch_ber = valid_running_ber / (iteration + 1)

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
    torch.save(model.state_dict(), './model/nn_keylen_' + str(args.keylen) + '.pth')
