clc
clear all
close all

config = set_config();

datafolder = 'data_experiment';
datasetname = 'received_data_40m_outdoor.mat';

message_ori_len = config.hide_msg_len;

% project_python_path = 'cd C:\Users\Guanxiong\OneDrive - 东南大学\RF_watermarking_project && C:/Users/Guanxiong/anaconda3/envs/torc_20230705/python.exe';
project_python_path = 'cd C:\Users\gxshe\OneDrive - 东南大学\RF_watermarking_project && C:/Users/gxshe/anaconda3/envs/torch/python.exe ';

% command = 'cd C:\Users\Guanxiong\OneDrive - 东南大学\RF_watermarking_project && C:/Users/Guanxiong/anaconda3/envs/torc_20230705/python.exe "./nn_extractor.py"';
% command = ['cd C:\Users\gxshe\OneDrive - 东南大学\RF_watermarking_project && C:/Users/gxshe/anaconda3/envs/torch/python.exe ' ...
%             './nn_extractor.py --keylen ' num2str(message_ori_len*3)];

command = [project_python_path ' ./nn_extractor.py --keylen ' num2str(message_ori_len*3) ' --bw ' num2str(config.lora.bw) ' --datafolder ' datafolder ' --datasetname ' datasetname];

status = system(command, '-echo');

load(['.\' datafolder '\' datasetname]);

load(['.\' datafolder '\message_nn_decoded_rx.mat'])

message_nn_decoded_rx = message_nn_decoded_rx';

num_pkts = size(message_nn_decoded_rx, 2);

bchenc = comm.BCHEncoder;
bchdec = comm.BCHDecoder;

% message_fec_decoded = zeros(45, 101);


ber_list_pre_fec = [];
per_list_pre_fec = [];

ber_list_post_fec = [];
per_list_post_fec = [];


message_padding_len = ceil(message_ori_len / 8) * 8 - message_ori_len;

for i = 1:num_pkts
    
    message_nn_decoded_tmp = message_nn_decoded_rx(:, i);
    message_deinterleaved_tmp = randdeintrlv(message_nn_decoded_tmp, 1997);
    % message_deinterleaved_tmp = matdeintrlv(message_nn_decoded_tmp, 10, 18);
    message_fec_decoded_tmp = bchdec(message_deinterleaved_tmp);
    

    message_payload_tmp = message_payload(1:end-message_padding_len, i);
    
    message_fec_encoded_tmp = bchenc(message_payload_tmp);
    % message_fec_encoded_tmp = matintrlv(message_fec_encoded_tmp, 10, 18);
    message_fec_encoded_tmp = randintrlv(message_fec_encoded_tmp, 1997);


    [ham_dis_pre_fec, ber_pre_fec] = biterr(message_fec_encoded_tmp, message_nn_decoded_tmp);
    [ham_dis_post_fec, ber_post_fec] = biterr(message_fec_decoded_tmp, message_payload_tmp);
    
    
    ber_list_pre_fec = [ber_list_pre_fec ber_pre_fec];
    per_list_pre_fec = [per_list_pre_fec ham_dis_pre_fec==0]; 
    
    ber_list_post_fec = [ber_list_post_fec ber_post_fec];
    per_list_post_fec = [per_list_post_fec ham_dis_post_fec==0]; 

end

ber_all_pre_fec = mean(ber_list_pre_fec);
per_all_pre_fec = length(find(per_list_pre_fec==0))/num_pkts;


ber_all_post_fec = mean(ber_list_post_fec);
per_all_post_fec = length(find(per_list_post_fec==0))/num_pkts;

fprintf(['Average SNR is ' num2str(mean(snr_rx)) ', Pre-FEC BER is ' num2str(ber_all_pre_fec) ', Post-FEC BER is ' num2str(ber_all_post_fec), ', PER is ' num2str(per_all_post_fec) '\n'])

