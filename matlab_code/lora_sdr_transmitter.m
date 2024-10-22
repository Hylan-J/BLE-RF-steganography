clc
clear all
close all


config = set_config();

loraphy = LoRaPHY(config.lora.rf_freq, config.lora.sf, config.lora.bw, config.lora.fs);
loraphy.has_header = 1;         % explicit header mode
loraphy.cr = 4;                 % code rate = 4/8 (1:4/5 2:4/6 3:4/7 4:4/8)
loraphy.crc = 1;                % enable payload CRC checksum
loraphy.preamble_len = 8;       % preamble: 8 basic upchirps

datafolder = 'data_experiment';
project_python_path = 'cd C:\Users\Guanxiong\OneDrive - 东南大学\RF_watermarking_project && C:/Users/Guanxiong/anaconda3/envs/torc_20230705/python.exe';

sdr_transmitter = sdr_initial(config, 'transmitter');

if strcmp(config.sdr.type, 'n210') || strcmp(config.sdr.type, 'b210')
    sdr_transmitter.EnableBurstMode = true;
    sdr_transmitter.NumFramesInBurst = 5;
end

% crcgenerator = comm.CRCGenerator([16 15 2 0]);

bchenc = comm.BCHEncoder;

pkt_transmit_total = 0;
num_pkt_group = 300;


message_ori_len = config.hide_msg_len; % 32-bit DevAddr  + 13-bit sequence number
message_padding_len = ceil(message_ori_len / 8) * 8 - message_ori_len;


while 1
    
    message_source_tx = zeros(message_ori_len, num_pkt_group);
    message_fec_encoded_tx = zeros(message_ori_len*(15/5), num_pkt_group);

    for pkt_ind = 1:num_pkt_group
        message_source_tx(:,pkt_ind) = randi([0 1], message_ori_len, 1);
        codeword_tmp = bchenc(message_source_tx(:,pkt_ind));
        % codeword_tmp = randintrlv(codeword_tmp, 4831);
        codeword_tmp = randintrlv(codeword_tmp, 1997);
        message_fec_encoded_tx(:,pkt_ind) = codeword_tmp;

        % message_fec_encoded_tx(:,pkt_ind) = bchenc(message_source_tx(:,pkt_ind));
    end
    
    save(['.\' datafolder '\message_fec_encoded_tx.mat'], 'message_fec_encoded_tx')

    % Generate 500 random transmited RF waveform by running a Python script
    % command = 'cd C:\Users\Guanxiong\OneDrive - 东南大学\RF_watermarking_project && C:/Users/Guanxiong/anaconda3/envs/torc_20230705/python.exe "./nn_embedder.py"';

    command = [project_python_path ' ./nn_embedder.py --keylen ' num2str(message_ori_len*3) ' --bw ' num2str(config.lora.bw) ' --datafolder ' datafolder];
    status = system(command, '-echo');

    load(['.\' datafolder '\preamble_tx.mat'], 'preamble_tx')
    
    % load('preamble_tx.mat', 'preamble_tx')

    % preamble_tx = h5read('data_experiment.h5','/preamble_tx');

    % preamble_tx = preamble_tx(:,1:size(preamble_tx,2)/2) + 1i*preamble_tx(:,size(preamble_tx,2)/2+1:end); % complex representation
    
    % message_m = message_source_tx;
    
    for ii = 1:size(preamble_tx, 1)

        % message_bit = [message_source_tx(:,ii)]; % append three additional zeros
        
        message_bit = [message_source_tx(:,ii);zeros(message_padding_len,1)]; % append three additional zeros

        message_transmit = bit2int(message_bit, 8);
        
        symbols = loraphy.encode(message_transmit);
        sig = loraphy.modulate(symbols);
        
        % Steganography
       
        preamble_stega = repmat(preamble_tx(ii,:), 1, loraphy.preamble_len-2);

        preamble_stega = preamble_stega/sqrt(mean(abs(preamble_stega).^2));

        sig(1:size(preamble_stega,2)) = preamble_stega;
        
        sig = sig/(2*abs(max([real(sig);imag(sig)])));
        sig = [sig;zeros(1000,1)];
        
        if abs(max([real(sig);imag(sig)]))>1
            fprintf('The amplitude of transmitted signal is too high! \n')
            continue
        end

        underrun = sdr_transmitter(sig);
        pkt_transmit_total = pkt_transmit_total + 1;
        
        if underrun == 0
            fprintf(['Transmitted LoRa packet index: ' num2str(pkt_transmit_total) '\n']);
        else
            fprintf('Underrun is detected! \n');
        end

        pause(1);

    end


end



