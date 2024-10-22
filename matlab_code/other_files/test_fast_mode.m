clc
clear all
close all

addpath('..');
rf_freq = 470e6;    % carrier frequency, used to correct clock drift
sf = 7;             % spreading factor
bw = 125e3;         % bandwidth
fs = 1e6;           % sampling rate

phy = LoRaPHY(rf_freq, sf, bw, fs);
phy.has_header = 1;         % explicit header mode
phy.cr = 4;                 % code rate = 4/8 (1:4/5 2:4/6 3:4/7 4:4/8)
phy.crc = 1;                % enable payload CRC checksum
phy.preamble_len = 8;       % preamble: 8 basic upchirps

% Encode payload [1 2 3 4 5]
% origin_data = (1:100)';
origin_data =  [2, 200]'; % range 0 - 255
symbols = phy.encode(origin_data);
% Baseband Modulation
sig = phy.modulate(symbols);


% stega = 0;
% if stega
%     load('data_encoded.mat')
%     preamble_stega = data_encoded(1,:);
% 
%     sig(1:8192) = repmat(preamble_stega, 1, 8);
% 
% end


snr_list = [0:5:10];
total_rounds = 50;

warning('off','all')

% Fast mode runs faster but has lower sensitivity
fprintf("Fast Mode\n");
phy.fast_mode = true;
ber_list1 = benchmark(phy, snr_list, sig, fs, bw, total_rounds, origin_data);

% Non-Fast mode runs slower but has higher sensitivity
% fprintf("Non-Fast Mode\n");
% phy.fast_mode = false;
% ber_list2 = benchmark(phy, snr_list, sig, fs, bw, total_rounds, origin_data);

markersize = 8;
linewidth = 3;
plot(snr_list, ber_list1, 'k', 'MarkerSize', markersize, 'LineWidth', linewidth);
hold on
% plot(snr_list, ber_list2, 'k--', 'MarkerSize', markersize, 'LineWidth', linewidth);
xlabel('SNR (dB)');
ylabel('Bit Error Rate');
set(gca, 'FontName', 'Arial', 'FontSize', 16, 'GridLineStyle', '--');
lgnd = legend('Fast Mode', 'Non-Fast Mode', 'Location', 'Northeast');
grid on

warning('on','all')

function ber_list = benchmark(phy, snr_list, sig, fs, bw, total_rounds, origin_data)
    tic
    ber_list = [];
    for snr = snr_list
        tmp_ber_list = zeros(total_rounds, 1);
        for ii = 1:total_rounds
            new_sig = awgn([zeros(100000,1);sig;zeros(5000,1);sig;zeros(5000,1);sig;zeros(5000,1)], snr-10*log10(fs/bw));
            % new_sig = new_sig(5600:end);
            % plot(real(new_sig))
            % new_sig = awgn([ones(500000,1)], snr-10*log10(fs/bw));
            try
                [symbols_d, ~, ~, ~] = phy.demodulate(new_sig);
                [data, checksum] = phy.decode(symbols_d);
                tmp_ber_list(ii) = calc_ber(data, origin_data);
            catch
                tmp_ber_list(ii) = 0.5;
            end
                
        end
        ber_list = [ber_list; mean(tmp_ber_list)];
    end
    toc
end

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
