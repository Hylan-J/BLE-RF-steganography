clc
clear all
close all

%% 

num_pkt = 100;
lora_ind = 1;
lora_type = 'EoRa';
sdr_ind = 1;


config = set_config();

% LoRa demodulation config

loraphy = LoRaPHY(config.lora.rf_freq, config.lora.sf, config.lora.bw, config.lora.fs);
loraphy.has_header = 1;         % explicit header mode
loraphy.cr = 4;                 % code rate = 4/8 (1:4/5 2:4/6 3:4/7 4:4/8)
loraphy.crc = 1;                % enable payload CRC checksum
loraphy.preamble_len = 8;       % preamble: 8 basic upchirps

datafolder = 'data_experiment';

% 
% message_ori_len = 40; % 32-bit DevAddr  + 13-bit sequence number
% message_padding_len = ceil(message_ori_len / 8) * 8 - message_ori_len;

%% 

sdr_receiver = sdr_initial(config, 'receiver'); % define SDR

pkt_idx = 0;

preamble_rx = [];
message_payload = [];
cfo_rx = [];
snr_rx = [];
timestamp = [];

while pkt_idx <= num_pkt
    
    [signal_sdr, ~, overrun] = sdr_receiver();
    
    try
        [symbols_d, preamble_phy, cfo_d, snr_d, ~] = loraphy.demodulate(signal_sdr);
    catch
        continue
    end

    [message_decoded, checksum] = loraphy.decode(symbols_d); % the last two are 

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
    
        pkt_idx = pkt_idx + num_frames;
    
        fprintf(['Info: ' num2str(size(preamble_rx,2)) ' LoRa packets are collected, estimated SNR is ' num2str(mean(snr_d)) ' dB \n'])

    end

    
end

release(sdr_receiver)

%% Save

save(['.\' datafolder '\received_data.mat'], 'preamble_rx', 'message_payload', 'cfo_rx', 'snr_rx')

% save('data_test.mat', 'preamble', 'message_source', 'cfo')

% preamble_iq = [real(preamble_rx);imag(preamble_rx)];
% 
% filename = 'data_experiment.h5';
% 
% h5create(filename, '/preamble_rx', size(preamble_rx));
% h5write(filename, '/preamble_rx', preamble_rx);
% 
% h5create(filename, '/message_payload', size(message_payload));
% h5write(filename, '/message_payload', message_payload);
% 
% h5create(filename, '/cfo', size(cfo_rx));
% h5write(filename, '/cfo', cfo_rx);
