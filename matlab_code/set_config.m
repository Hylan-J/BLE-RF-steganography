function config = set_config()

    % LoRa config
    config.lora.rf_freq = 433e6;    % carrier frequency, used to correct clock drift
    % config.lora.rf_freq = 2.472e9;    % carrier frequency, used to correct clock drift
    config.lora.sf = 7;             % spreading factor
    config.lora.fs = 1e6;           % sampling rate
    
    config.lora.bw = 500e3;         % bandwidth
    config.hide_msg_len = 40;
    
    % SDR config
    config.sdr.type = 'b210';
    config.sdr.fc = config.lora.rf_freq; % Center frequency (Hz)
    config.sdr.FrontEndSampleRate = config.lora.fs;     % Samples per second
    config.sdr.FrameLength = 100000;  % Frame length 375000
    config.sdr.idx = 1;

    if strcmp(config.sdr.type, 'b210') 
        config.sdr.tx_gain = 70; % b210 maximum 89 db
        config.sdr.rx_gain = 70; % b210 maximum 76 db
    else 
        config.sdr.tx_gain = 0; % avaiable range -89.75 to 0
        config.sdr.rx_gain = 70; % avaiable range -4 to 71
    end

end