function config = set_config_simulation()

    % LoRa config
    config.lora.rf_freq = 433e6;    % carrier frequency, used to correct clock drift
    % config.lora.rf_freq = 2.472e9;    % carrier frequency, used to correct clock drift
    config.lora.sf = 7;             % spreading factor
    config.lora.fs = 1e6;           % sampling rate
    
    config.lora.bw = 500e3;         % bandwidth
    config.hide_msg_len = 60;
    
    % SDR config
    % config.sdr.type = 'b210';
    % config.sdr.fc = config.lora.rf_freq; % Center frequency (Hz)
    % config.sdr.FrontEndSampleRate = config.lora.fs;     % Samples per second
    % config.sdr.FrameLength = 100000;  % Frame length 375000
    % config.sdr.idx = 1;
    % config.sdr.tx_gain = 70; % 10 for n210, 0 for pluto, 40 for b210
    % config.sdr.rx_gain = 80; % 30 for pluto, 40 for b210

end