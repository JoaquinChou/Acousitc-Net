% 保存未添加白噪声时的信号，以便后面画图观看两个信号差别
p_temp = p;

% 加入白噪声
SNR = 10;  %噪声信噪比
for i = 1:size(p,1)
    p(i,:) = awgn(p(i,:), SNR);
end

figure; plot(1:length(p(1,:)),p(1,:));
hold on; plot(1:length(p(1,:)),p_temp(1,:));

fprintf('\tAdd white noise with a signal-to-noise ratio of %f...\n', SNR);