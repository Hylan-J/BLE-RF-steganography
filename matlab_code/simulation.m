clc
clear all
close all


config = set_config_simulation();

loraphy = LoRaPHY(config.lora.rf_freq, config.lora.sf, config.lora.bw, config.lora.fs);
loraphy.has_header = 1;         % explicit header mode
loraphy.cr = 4;                 % code rate = 4/8 (1:4/5 2:4/6 3:4/7 4:4/8)
loraphy.crc = 1;                % enable payload CRC checksum
loraphy.preamble_len = 8;       % preamble: 8 basic upchirps


datafolder = 'data_simulation';
project_python_path = 'cd C:\Users\gxshe\OneDrive - 东南大学\RF_watermarking_project && C:/Users/gxshe/anaconda3/envs/torch/python.exe ';
% project_python_path = 'cd C:\Users\Guanxiong\OneDrive - 东南大学\RF_watermarking_project && C:/Users/Guanxiong/anaconda3/envs/torc_20230705/python.exe';


bchenc = comm.BCHEncoder;

num_pkts = 1000;

message_ori_len = config.hide_msg_len;  


%% Generate messages and distorted preamble

message_source_tx = zeros(message_ori_len, num_pkts);
message_fec_encoded_tx = zeros(message_ori_len*(15/5), num_pkts);


for pkt_ind = 1:num_pkts

    message_source_tx(:,pkt_ind) = randi([0 1], message_ori_len, 1);
    codeword_tmp = bchenc(message_source_tx(:,pkt_ind));
    codeword_tmp = randintrlv(codeword_tmp, 1997);
    % codeword_tmp = matintrlv(codeword_tmp, 10, 18);
    message_fec_encoded_tx(:,pkt_ind) = codeword_tmp;

end

save(['.\' datafolder '\message_fec_encoded_tx.mat'], 'message_fec_encoded_tx')

% command = 'cd C:\Users\Guanxiong\OneDrive - 东南大学\RF_watermarking_project && C:/Users/Guanxiong/anaconda3/envs/torc_20230705/python.exe "./nn_embedder.py"';
command = [project_python_path ' ./nn_embedder.py --keylen ' num2str(message_ori_len*3) ' --bw ' num2str(config.lora.bw) ' --datafolder ' datafolder];

status = system(command, '-echo');

load(['.\' datafolder '\preamble_tx.mat'], 'preamble_tx')


%%

preamble_rx = [];
message_payload = [];
cfo_rx = [];
snr_rx = [];
timestamp = [];

message_padding_len = ceil(message_ori_len / 8) * 8 - message_ori_len;

for ii = 1:num_pkts

    % message_bit = [message_source_tx(:,ii)];
    
    message_bit = [message_source_tx(:,ii);zeros(message_padding_len,1)]; % append three additional zeros

    message_transmit = bit2int(message_bit, 8);

    symbols = loraphy.encode(message_transmit);
    sig = loraphy.modulate(symbols);
    
    % Steganography
   
    preamble_stega = repmat(preamble_tx(ii,:), 1, loraphy.preamble_len);

    preamble_stega = preamble_stega/sqrt(mean(abs(preamble_stega).^2));
    
    % plot(real(preamble_stega(1:256)))


    sig(1:size(preamble_stega,2)) = preamble_stega;
    % sig(1:1024) = preamble_m(i,:);
    
    sig = sig/(2*abs(max([real(sig);imag(sig)])));
    sig = [zeros(1000,1);sig;zeros(1000,1)];

    % sig = iqimbal(sig, gen_rand(-1,1),  gen_rand(-5,5));
    % 
    % saleh = comm.MemorylessNonlinearity('Method','Saleh model', ...
    %         'AMAMParameters', [gen_rand(2.1587*0.95, 2.1587*1.05) gen_rand(1.1517*0.95, 1.1517*1.05)], ...
    %         'AMPMParameters', [gen_rand(4.0033*0.95, 4.0033*1.05) gen_rand(9.1040*0.95, 9.1040*1.05)]);
    % 
    % sig = saleh(sig);
    % % 
    phznoise = comm.PhaseNoise('SampleRate', config.lora.fs, 'Level',-50, 'FrequencyOffset',20);
    sig = phznoise(sig);
    
    % sig = frequencyOffset(sig, config.lora.fs, gen_rand(-200, 200));

    sig = awgn(sig, 30, 'measured');

    [symbols_d, preamble_phy, cfo_d, snr_d, ~] = loraphy.demodulate(sig);
    
    [message_decoded, checksum] = loraphy.decode(symbols_d);

    if ~isempty(message_decoded) && mean(checksum)==1

        message_decoded = message_decoded(1:length(message_decoded)-2,:); % remove CRC
        
        message_bit = int2bit(message_decoded, 8);

        num_frames = size(symbols_d, 2);
        timestamp_pkt = datetime('now');
        
        preamble_rx = [preamble_rx preamble_phy];
        message_payload = [message_payload message_bit];
        cfo_rx = [cfo_rx cfo_d];
        snr_rx = [snr_rx snr_d];
        timestamp = [timestamp timestamp_pkt];
    
    
        fprintf(['Info: ' num2str(size(preamble_rx,2)) ' LoRa packets are collected, estimated SNR is ' num2str(mean(snr_d)) ' dB \n'])
        
    end
end

save(['.\' datafolder '\received_data.mat'], 'preamble_rx', 'message_payload', 'cfo_rx', 'snr_rx')


%% Parse data


% command = 'cd C:\Users\Guanxiong\OneDrive - 东南大学\RF_watermarking_project && C:/Users/Guanxiong/anaconda3/envs/torc_20230705/python.exe "./nn_extractor.py"';
command = [project_python_path ' ./nn_extractor.py --keylen ' num2str(message_ori_len*3) ' --bw ' num2str(config.lora.bw) ' --datafolder ' datafolder];

status = system(command, '-echo');

load(['.\' datafolder '\received_data.mat']);

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

fprintf(['Pre-FEC BER is ' num2str(ber_all_pre_fec) ', Post-FEC BER is ' num2str(ber_all_post_fec), ', PER is ' num2str(per_all_post_fec) '\n'])



function out = gen_rand(min_val, max_val)

out = (max_val-min_val).*rand(1,1) + min_val;

end


% function sig_out = hardware_impairment(sig_in)
% 
% amp_imb = 1;
% phase_imb = 4;
% 
% amam_para = [2.1587 1.1517];
% ampm_para = [4.0033 9.1040];
% 
% 
% saleh = comm.MemorylessNonlinearity('Method','Saleh model');
% 
% sig_in = iqimbal(sig_in, amp_imb, phase_imb);
% 
% sig_in = saleh(sig_in);
% 
% sig_out = sig_in;
% 
% 
% end



function [sig_out, myPathGain] = augmentation(sig_in, Ts)
    
    sig_len = length(sig_in);

    t_rms_min = 10;
    t_rms_max = 100;
    t_rms = ((t_rms_max-t_rms_min).*rand(1) + t_rms_min)*1e-9; % random RMS delay spread from 10 to 300 ns.

    % t_rms = 100*1e-9; % random RMS delay spread from 10 to 300 ns.
    [avgPathGains, pathDelays]= exp_PDP(t_rms, Ts);
    
    %                 wavelength = 3e8/868.1e6;
    %                 speed = (5-0).*rand(1) + 0;
    %                 fD = speed/wavelength;
    %                 fD = (10-0).*rand(1) + 0;
    
    % fD = (10-0).*rand(1) + 0;
    fD = 0;
    % k_factor = (10-0).*rand(1) + 0;
    
    % wirelessChan = comm.RicianChannel('SampleRate',1/Ts,'KFactor',k_factor,'MaximumDopplerShift',fD,...
    %     'PathDelays',pathDelays,'AveragePathGains',avgPathGains,'DopplerSpectrum', doppler('Jakes'),...
    %     'PathGainsOutputPort',true);
    
    wirelessChan = comm.RayleighChannel('SampleRate',1/Ts,'MaximumDopplerShift',fD,...
    'PathDelays',pathDelays,'AveragePathGains',avgPathGains,'DopplerSpectrum', doppler('Jakes'),...
    'PathGainsOutputPort',true);

    chanInfo = info(wirelessChan);
    delay = chanInfo.ChannelFilterDelay;
    
    chInput = [sig_in;zeros(50,1)];
    [chOut, myPathGain] = wirelessChan(chInput);
    sig_out = chOut(delay+1:sig_len+delay);
    
end

function [avgPathGains,pathDelays ]=exp_PDP(tau_d,Ts)

A_dB = -30;


sigma_tau = tau_d; 
A=10^(A_dB/10);
lmax=ceil(-tau_d*log(A)/Ts); % Eq.(2.2)

% Exponential PDP
p=0:lmax; 
pathDelays = p*Ts;


p = (1/sigma_tau)*exp(-p*Ts/sigma_tau);
p_norm = p/sum(p);


avgPathGains = 10*log10(p_norm); % convert to dB

end

% function plot_wave(input)
% 
% 
% 
% 
% 
% end