#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description: 比特嵌入器、比特提取器网络部分
import torch.nn as nn
import torch.optim

from utils import split_complex


class ConvBlock(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """
    def __init__(self, channels_in, channels_out):
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(channels_in, channels_out, 7, padding='same'),
            nn.GELU()
        )

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, key_len, num_channels=256, num_layers=8):
        super(Encoder, self).__init__()

        layers = [ConvBlock(2 + key_len, num_channels)]
        for _ in range(int(num_layers / 2) - 1):
            layers.append(ConvBlock(num_channels, num_channels))
        self.conv_bns = nn.Sequential(*layers)

        layers_2 = [ConvBlock(num_channels + 2 + key_len, num_channels)]
        for _ in range(int(num_layers / 2) - 1):
            layers_2.append(ConvBlock(num_channels, num_channels))

        self.after_concat_layer = nn.Sequential(*layers_2)
        self.final_layer = nn.Conv1d(num_channels + 2 + key_len, 2, kernel_size=1)

    def forward(self, secret, cover):
        """
        secret：密码
        cover：原始信号
        """
        cover_split = split_complex(cover)
        secret = secret.unsqueeze(-1)
        secret_expand = secret.expand(-1, -1, cover.size(-1))
        encoded_cover = self.conv_bns(torch.cat([cover_split, secret_expand], dim=1))
        container = self.after_concat_layer(torch.cat([encoded_cover, cover_split, secret_expand], dim=1))
        container = self.final_layer(torch.cat([container, cover_split, secret_expand], dim=1))
        return container


class Decoder(nn.Module):
    def __init__(self, key_len, num_channels=256, num_layers=8):
        super(Decoder, self).__init__()

        layers = [ConvBlock(2, num_channels)]
        for _ in range(num_layers - 1):
            layers.append(ConvBlock(num_channels, num_channels))
        layers.append(ConvBlock(num_channels, key_len))
        layers.append(nn.AdaptiveAvgPool1d(output_size=1))

        self.layers = nn.Sequential(*layers)
        self.linear = nn.Linear(key_len, key_len)

    def forward(self, x):
        x = split_complex(x)
        x = self.layers(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.linear(x)

        return x


class StegNet(nn.Module):
    def __init__(self, key_len):
        super(StegNet, self).__init__()
        self.encoder = Encoder(key_len)
        self.decoder = Decoder(key_len)

    def encode(self, secret, cover):
        container = self.encoder(secret, cover)
        container = torch.complex(container[:, 0, :], container[:, 1, :])
        return container

    def decode(self, container_noisy):
        factor = torch.sqrt(torch.mean(torch.abs(container_noisy) ** 2))
        container_noisy = container_noisy / factor.item()
        secret_restored = self.decoder(container_noisy)
        return secret_restored
