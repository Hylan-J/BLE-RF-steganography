import argparse
import time

import numpy as np
import torch
from scipy.io import savemat, loadmat

from model import StegNet
from utils import lora_dataset


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--keylen', type=int, default=180, help='key size')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    parser.add_argument('--datafolder', type=str, default='data_simulation', help='path to save the data')

    parser.add_argument('--fs', type=float, default=1e6, help='sampling frequency')
    parser.add_argument('--sf', type=int, default=7, help='spreading factor')
    parser.add_argument('--bw', type=float, default=500e3, help='bandwidth')

    args = parser.parse_args()

    return args


def gen_tx_sig(args):
    """
    生成发送器的信号
    :param args:
    :param num_pkts:
    :return:
    """
    # 加载发射机的隐蔽消息 FEC 编码后的数据
    f = loadmat('./matlab_code/' + args.datafolder + '/message_fec_encoded_tx.mat')
    message_fec_encoded = f['message_fec_encoded_tx']
    print(message_fec_encoded.shape)
    # 获取隐蔽消息的条数
    num_pkts = message_fec_encoded.shape[1]
    print(num_pkts)
    for i in range(num_pkts):
        message_tmp = message_fec_encoded[:, i]
        print(message_tmp)
        print(message_tmp.shape)
        message_tmp = np.expand_dims(message_tmp, axis=0)

    print('TX signal saved, current time is ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


if __name__ == '__main__':
    args = args_parser()
    gen_tx_sig(args)
