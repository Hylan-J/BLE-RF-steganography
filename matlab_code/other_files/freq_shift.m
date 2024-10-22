function sig_shift = freq_shift(sig_in, freq, Ts)

    n = [1:length(sig_in)]';
    sig_shift = sig_in.*exp(1i*2*pi*freq*n*Ts);

end