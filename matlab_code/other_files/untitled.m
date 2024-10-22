
clc
clear all
close all
% 
% enc = comm.BCHEncoder;
% mod = comm.DPSKModulator(BitInput=true);
% snr = 10;
% demod = comm.DPSKDemodulator(BitOutput=true);
% dec = comm.BCHDecoder;
% errorRate = comm.ErrorRate(ComputationDelay=3);
% 
% for counter = 1:20
%   data = randi([0 1],30,1);
%   encodedData = enc(data);
%   modSignal = mod(encodedData);
%   receivedSignal = awgn(modSignal,snr);
%   demodSignal = demod(receivedSignal);
%   receivedBits = dec(demodSignal);
%   errorStats = errorRate(data,receivedBits);
% end
% fprintf('Error rate = %f\nNumber of errors = %d\n', ...
%   errorStats(1),errorStats(2))


% num_pkt_group = 100;
% bch = comm.BCHEncoder;
% 
% message_fec_encoded = zeros(80*(15/5), num_pkt_group);
% 
% for pkt_ind = 1:num_pkt_group
%     message_source = randi([0 1],80,1);
%     message_fec_encoded(:,pkt_ind) = bch(message_source);
% end
% 
% save('tx_message.mat', 'message_fec_encoded')



% t = bchnumerr(15, 5);
