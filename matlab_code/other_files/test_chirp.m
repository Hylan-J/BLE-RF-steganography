clc
clear all
close all

load upchirp.mat

fs = 1e6;
Ts = 1/fs;

n = [1:length(upchirp)]';

f = 10000;

sig = 0.2*exp(1i*2*pi*f*n*Ts);
% 
figure
plot(real(sig + upchirp))

figure
plot(abs(fftshift(fft((sig+upchirp)))))

% plot(real(sig))
% hold on 
% plot(real(upchirp))