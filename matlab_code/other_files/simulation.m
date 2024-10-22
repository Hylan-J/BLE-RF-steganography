clc
clear all
close all

addpath('..');

rf_freq = 470e6;    % carrier frequency, used to correct clock drift
sf = 7;             % spreading factor
bw = 125e3;         % bandwidth
fs = 1e6;           % sampling rate

loraphy = LoRaPHY(rf_freq, sf, bw, fs);
loraphy.has_header = 1;         % explicit header mode
loraphy.cr = 3;                 % code rate = 4/8 (1:4/5 2:4/6 3:4/7 4:4/8)
loraphy.crc = 1;                % enable payload CRC checksum
loraphy.preamble_len = 8;       % preamble: 8 basic upchirps


data_m = h5read('tx_sig.h5','/preamble')';
message_m = h5read('tx_sig.h5','/message')';

data_complex_m = data_m(:,1:size(data_m,2)/2) + 1i*data_m(:,size(data_m,2)/2+1:end);

% num_pkts = size(data_m, 1);

num_pkts = 100;

tmp_ber_list = zeros(num_pkts, 1);

sig_tx = [];

for i = 1:num_pkts

    message_bit = message_m(i,:);
    message_transmit = bit2int(message_bit', 8);
    
    symbols = loraphy.encode(message_transmit);
    sig = loraphy.modulate(symbols);
    
    % Steganography

    preamble_stega = repmat(data_complex_m(i,:), 1, loraphy.preamble_len-2);
    preamble_stega = preamble_stega/sqrt(mean(abs(preamble_stega).^2));
    sig(1:size(preamble_stega,2)) = preamble_stega;

    plot(abs(fftshift(fft(sig(1:2048)))));


    sig_tx = [sig_tx sig];

    fprintf(['Generate LoRa packet, index: ' num2str(i) '\n']);
end



preamble_rx = [];
message_rx = [];
cfo_est = [];

snr = 10;

for i = 1:num_pkts
    
    sig_rx = sig_tx(:,i);

    cfo_shift = rand_in_range(-5000, 5000);
    % cfo_shift = 1000;
    sig_rx = freq_shift(sig_rx, cfo_shift, 1/fs);

    sig_rx = awgn([zeros(3000,1);sig_rx;zeros(3000,1)], snr, 'measured');
    % sig_rx = awgn([ones(100000,1)], snr-10*log10(fs/bw));

    % try
    [symbols_d, preamble_complex, cfo_d, ~] = loraphy.demodulate(sig_rx);
    
    freq_error = abs(cfo_d -cfo_shift);

    preamble_iq = [real(preamble_complex);imag(preamble_complex)];

    [message_received, checksum] = loraphy.decode(symbols_d);
    
    message_received = message_received(1:length(message_received)-2); % remove CRC
    
    message_bit = int2bit(message_received, 8);
    
    % message_bit(message_bit==0) = -1;

    tmp_ber_list(i) = calc_ber(message_received, message_transmit);

    preamble_rx = [preamble_rx preamble_iq];
    message_rx = [message_rx message_bit];
    cfo_est = [cfo_est cfo_d];
    fprintf(['Decode LoRa packet, index: ' num2str(i) '\n']);
        
    % catch
    %     tmp_ber_list(i) = 0.5;
    % end

end

ber_avg = mean(tmp_ber_list);


filename = 'rx_sig.h5';

h5create(filename, '/preamble', size(preamble_rx));
h5write(filename, '/preamble', preamble_rx);

h5create(filename, '/message', size(message_rx));
h5write(filename, '/message', message_rx);

h5create(filename, '/cfo', size(cfo_est));
h5write(filename, '/cfo', cfo_est);


% function sig_shift = freq_shift(sig_in, freq, Ts)
% 
%     n = [1:length(sig_in)]';
%     sig_shift = sig_in.*exp(1i*2*pi*freq*n*Ts);
% 
% end

function ber = calc_ber(res, ground_truth)
    len1 = length(res);
    len2 = length(ground_truth);
    len = min(len1, len2);
    if len > 0
        res = res(1:len);
        ground_truth = ground_truth(1:len);
        err_bits = sum(xor(de2bi(res, 8), de2bi(ground_truth, 8)), 'all') + 8 * (len2 - len);
        ber = err_bits / (len2 * 8);
    else
        ber = 0.5;
    end
end

function r = rand_in_range(a, b)

    r = (b-a).*rand(1,1) + a;

end

% function message_byte = bit2byte(message_bit)
% 
%     message_byte = zeros(length(message_bit)/8, 8);
%     num_byte = size(message_byte, 1);
% 
%     for ii = 1:num_byte
% 
%         message_byte(ii,:) = message_bit((ii-1)*8 + 1: ii*8);
% 
%     end
% 
% end
