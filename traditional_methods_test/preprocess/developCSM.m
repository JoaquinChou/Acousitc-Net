function [CSM, freqs] = developCSM(p, freq_l, freq_u, Fs, t_start, t_end)
% Develop CSM from microphone array measurements. Pressure time signals.
% Creates time blocks and averages to reduce noise for CSM given your
% frequency range with possible overlap to generate more blocks. See
% Welch's method for more info.
%
% p:麦克风阵列输入 n*N_mic


% 采样点个数
N_total_samples = size(p, 1);

% 麦克风阵列数
N_signals = size(p, 2);

% 采样时间
t_signal = N_total_samples/Fs;


if nargin < 6
    t_start = 0;      % 默认0时刻开始
    t_end = t_signal; % 持续时间后结束
end
if (t_start < 0) || (t_end > t_signal) || (t_signal>(t_end-t_start))
    error('Time-boundaries out of bounds!');
end


% 开始和结束的时间点
start_sample = floor(t_start*Fs) + 1;
block_samples = ceil(t_signal*Fs);  %采样点数              
x_fr = Fs / block_samples * (0:floor(block_samples/2)-1);

% 选取在频率之间的点
freq_sels = find((x_fr>=freq_l).*(x_fr<=freq_u));
N_freqs = length(freq_sels);

CSM = zeros(N_signals, N_signals, N_freqs);

N_start = start_sample;
N_end = N_start + block_samples - 1;
p_fft = 2*fft(p(N_start:N_end,:))/block_samples;

for K = 1:N_freqs
    CSM(:,:,K) = CSM(:,:,K) + 0.5*p_fft(freq_sels(K),:)'*p_fft(freq_sels(K),:);
%     CSM(:,:,K) = CSM(:,:,K)-diag(diag(CSM(:,:,K))-diag(0));
end
    

freqs = x_fr(freq_sels);

end