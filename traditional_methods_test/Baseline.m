clear; clc; close all;

count = 0;
Fs = 51200;
z_source = 2.5;

source_path = 'E:/研究生/实验室工作/涂老师/声学会议/traditional_methods_test/data';
source_dir = dir(source_path);

result = zeros(1000,2);

tic;
% for i = 1:length(source_dir)-2
for i = 1:1
    data = h5read([source_path,'/', source_dir(i+2).name],'/time_data');
%     
    for j = 1:1
        
        plot(data(j,:));
%         axis off;
        xlim([0 1024]);
        title(['Original Wave'],'fontsize',20);
        ylabel('amplitude ','fontsize',18)
        xlabel('time (s)','fontsize',18)
        hold on;
    end
   
    % 提取真实标签
%     Real = regexp(source_dir(i+2).name, '_', 'split');
%     x_Real_cell = Real(2); y_Real_cell = Real(4); rms_Real_cell = Real(6);
%     x_Real_str = x_Real_cell{1}; y_Real_str = y_Real_cell{1}; rms_Real_str = rms_Real_cell{1};
%     x_Real = str2double(x_Real_str); y_Real = str2double(y_Real_str); 
%     
%     [X,Y,SPL,Mic_ac] = Experiment_SourcesAndMics(data, Fs, z_source, x_Real, y_Real);
%     
%     % 提取真实SPL
%     distance = norm(Mic_ac - [x_Real,y_Real,z_source]);
%     rms_Real = 20*log10(real(str2double(rms_Real_str))/distance/distance/2e-5);
%     
%     
%     
%     % DAS或其他算法预测的标签
%     [y,x] = find(SPL==max(max(SPL)));
%     x_pos = X(x); y_pos = Y(y); rms = max(max(SPL));
%     plot(x_pos, y_pos, 'x', 'MarkerSize',7,'color','b');
% %     axes('Position',[0.18,0.62,0.28,0.25]);
%     % 预测误差
%     pos_real = [x_Real,y_Real]; pos_est = [x_pos,y_pos];
%     pos_err = norm(pos_real-pos_est)/(norm(pos_real)+eps);
%     rms_err = abs(rms_Real-rms)/(abs(rms_Real)+eps);
%     
%     % 保存
%     result(i,1) = pos_err;
%     result(i,2) = rms_err;
%    
    % 计数
    count = count + 1
end
time = toc;

average_err = mean(result);




