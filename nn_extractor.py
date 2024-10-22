import argparse

import numpy as np
import torch
from scipy import stats
from scipy.io import savemat, loadmat

from model import StegNet
from utils import lora_dataset, cfo_compensation


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--keylen', type=int, default=120, help='key size')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    parser.add_argument('--datafolder', type=str, default='data_simulation', help='path to save the data')
    parser.add_argument('--datasetname', type=str, default='received_data.mat', help='name of the dataset')

    parser.add_argument('--fs', type=float, default=1e6, help='sampling frequency')
    parser.add_argument('--sf', type=int, default=7, help='spreading factor')
    parser.add_argument('--bw', type=float, default=500e3, help='bandwidth')

    args = parser.parse_args()

    return args


def parse_rx_sig(args):
    file_content = loadmat('./matlab_code/' + args.datafolder + '/' + args.datasetname)
    data_complex = file_content['preamble_rx']
    cfo_rx = file_content['cfo_rx']
    message_payload = file_content['message_payload']

    sig_len = int(data_complex.shape[0])

    # sig_len = int(data.shape[0]/2)
    # data_complex = data[:sig_len, :] + 1j*data[sig_len:, :]
    # del data

    num_pkts = data_complex.shape[1]
    sig_len = int(data_complex.shape[0])

    std_chirp = lora_dataset(sf=args.sf, bw=args.bw, fs=args.fs, num_symbols=1, num_packets=1)
    std_chirp = np.expand_dims(std_chirp[0], axis=0)
    std_chirp = torch.from_numpy(std_chirp.astype(np.complex64)).to(args.device)

    for i in range(num_pkts):
        data_complex[:, i] = cfo_compensation(data_complex[:, i], cfo_rx[:, i], 1 / args.fs)

    model = StegNet(key_len=args.keylen).to(args.device)
    model.load_state_dict(
        torch.load('./model/nn_keylen_' + str(args.keylen) + '_bw_' + str(int(args.bw / 1e3)) + '.pth'))
    model.eval()

    print('Number of packets to parse: ', num_pkts)

    message_nn_decoded_m = []

    for i in range(num_pkts):
        preamble = data_complex[:, i]
        # preamble = preamble/np.sqrt(np.mean(np.abs(preamble)**2)) # normalize the preamble

        preamble = np.expand_dims(preamble, axis=0)
        preamble = torch.from_numpy(preamble.astype(np.complex64)).to(args.device)

        rep_decode = True
        if rep_decode:

            num_stega_preamble = 8
            message_decoded_m = np.zeros((num_stega_preamble, args.keylen))

            for preamble_ind in range(num_stega_preamble):
                message_decoded = model.decode(
                    preamble[:, int((sig_len / 8) * (preamble_ind)):int((sig_len / 8) * (preamble_ind + 1))])
                message_decoded = torch.sign(message_decoded)
                message_decoded = message_decoded.cpu().detach().numpy()

                message_decoded_m[preamble_ind] = message_decoded

            message_nn_decoded = stats.mode(message_decoded_m, keepdims=True)[0]

            # message_nn_decoded = np.mean(message_decoded_m, axis = 0)
            # message_nn_decoded = np.sign(message_nn_decoded)

            message_nn_decoded = message_nn_decoded.astype(int).squeeze()
            message_nn_decoded[message_nn_decoded == -1] = 0
            message_nn_decoded_m.append(message_nn_decoded)

        else:

            message_nn_decoded = model.decode(preamble[:, 2 * int(sig_len / 8):3 * int(sig_len / 8)])
            # message_nn_decoded = model.decode(preamble[:int(sig_len/8)])
            # message_nn_decoded = model.decode(std_container)
            message_nn_decoded = torch.sign(message_nn_decoded)

            message_nn_decoded = message_nn_decoded.cpu().detach().numpy().astype(int).squeeze()
            message_nn_decoded[message_nn_decoded == -1] = 0
            message_nn_decoded_m.append(message_nn_decoded)

    # mdic = {"message_nn_decoded_rx": message_nn_decoded_m,
    #         "message_payload": message_payload,
    #         "cfo_rx": cfo_rx,
    #         "preamble_rx": data_complex}

    # savemat('./sdr_code/data_experiment.mat', mdic)

    savemat('./matlab_code/' + args.datafolder + '/message_nn_decoded_rx.mat',
            {"message_nn_decoded_rx": message_nn_decoded_m})


if __name__ == '__main__':
    args = args_parser()
    parse_rx_sig(args)
