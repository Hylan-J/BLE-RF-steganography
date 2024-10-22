#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description: 比特嵌入器操作部分
import argparse
import time

import numpy as np
import torch
from scipy.io import savemat, loadmat

from model import StegNet
from utils import lora_dataset


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--keylen', type=int, default=20, help='key size')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    parser.add_argument('--datafolder', type=str, default='data_simulation', help='path to save the data')

    args = parser.parse_args()

    return args


def gen_tx_sig(args):
    """
    生成发送器的信号
    :param args:
    :param num_pkts:
    :return:
    """
    # 实例化编码器模型
    model = StegNet(key_len=args.keylen).to(args.device)
    # 模型加载参数
    model.load_state_dict(
        torch.load('./model/nn_keylen_' + str(args.keylen) + '_bw_' + str(int(args.bw / 1e3)) + '.pth'))
    # 模型开启评估模式
    model.eval()

    std_chirp = lora_dataset(sf=args.sf, fs=args.fs, bw=args.bw, num_symbols=1, num_packets=1)
    std_chirp = np.expand_dims(std_chirp[0], axis=0)
    std_chirp = torch.from_numpy(std_chirp.astype(np.complex64)).to(args.device)

    # 加载发射机的隐蔽消息 FEC 编码后的数据
    f = loadmat('./matlab_code/' + args.datafolder + '/message_fec_encoded_tx.mat')
    message_fec_encoded = f['message_fec_encoded_tx']
    # 获取隐蔽消息的条数
    num_pkts = message_fec_encoded.shape[1]
    # 生成的波形
    wave_m = np.zeros((num_pkts, std_chirp.shape[1]), dtype=np.complex64)

    for i in range(num_pkts):
        # 获取第i个FEC编码后的隐蔽消息(shape:(180,))
        message_tmp = message_fec_encoded[:, i]
        message_tmp = np.expand_dims(message_tmp, axis=0)
        message_tmp = torch.from_numpy(message_tmp.astype(np.float32)).to(args.device)
        message_tmp[message_tmp == 0] = -1
        # 生成隐蔽后的前导码波形数据
        container = model.encode(message_tmp, std_chirp)
        wave_m[i] = container.cpu().detach().numpy()

    # 隐藏后的波形数据
    mdic = {"preamble_tx": wave_m}
    # 保存数据
    savemat('./matlab_code/' + args.datafolder + '/preamble_tx.mat', mdic)

    print('TX signal saved, current time is ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


if __name__ == '__main__':
    args = args_parser()
    gen_tx_sig(args)
