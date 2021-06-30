function [p, Fs] = simulateArraydata(source_info, mic_info, c, Fs, duration)
%simulateArraydata   模拟麦克风阵列的压力数据。
%   simulateArraydata(source_info, mic_info, c, Fs, duration) 
%   生成仿真麦克风阵列数据
%   
%   simulateArraydata(source_info, mic_info) minimum requirement for call
%   req SPL_at_array] for each row, 
%   i.e. several rows are more than one source. This generates tonal noise
%   source for frequency freq. If freq = 0 will output white noise.
%   Multiple white noise sources currently not working
%
%   source_info(l, :) is [x y z f]


N_source = size(source_info, 1);  %声源个数
N_mic = size(mic_info, 1);   %麦克风个数
f_min = min(source_info(:, 4));  %声源最低频率
f_max = max(source_info(:, 4));  %声源最高频率

% 阵列中心
x_ac = mean(mic_info,1);

% 输入参数小于3时，设置默认声速
if nargin < 3
    % 默认声速（m/s）
    c = 343;
end

if nargin < 4
    % 将采样频率设置为声源最高频率的20倍
    Fs = 20*f_max;   
end

if nargin < 5
    % 将信号持续时间设置为声源周期的10倍
    duration = 10/f_min;  
end

% 计算样本点个数（分帧）
t = 0:1/Fs:(duration-1/Fs);
N_samples = length(t);

% 麦克风阵列数据初始化
p = zeros(N_mic, N_samples);

% 对每个声源累加声音信号
for I = 1:N_source
    % 阵列中心到某个声源距离
    r_ac = norm(x_ac-source_info(I, 1:3));
    
    % 声压级 SPL = 20*log10(amp/2e-5), we scale by the distance from
    % source to array center to obtain given SPL at center.
    amp = r_ac*2e-5*10^(source_info(I, 5)/20);

    for J = 1:N_mic

        r = sqrt(dot(mic_info(J, :) - source_info(I, 1:3), mic_info(J, :) - source_info(I, 1:3)) );

        % 延迟时间
        ph = r/ c;

        p(J, :) = p(J, :) + sqrt(2)*amp*cos(2*pi*source_info(I, 4)*(t-ph))/r;
        % Simulate dipole, still needs works
        % p(J, :) = p(J, :) + (0.0001*(2*pi*source_info(I, 4)/c)^2*sin(acos(source_info(I, 3)/r)))*sqrt(2)*amp*cos(2*pi*source_info(I, 4)*(t-ph))/r;
    end 
end    

end