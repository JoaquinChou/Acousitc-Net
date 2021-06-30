clear; clc; close all;
figure;
% 此处以x_-0.03_y_-0.21_rms_0.64_sources.h5为例子
x_Real = -0.03;
y_Real = -0.21;
% 由Acoustic-Net预测的结果为
x_Predict = -0.03
y_Predict = -0.22

plot(x_Real, y_Real, 'p','MarkerSize',7,'color','k');
hold on;
plot(x_Predict, y_Predict, 'x','MarkerSize',7,'color','b');

xlim([-1.5 1.5]);
ylim([-1.5 1.5])
title(['Acoustic-Net'],'fontname','Times New Roman','fontsize',18,'FontWeight','bold');
xlabel('x (m)','fontsize',18,'fontname','Times New Roman','FontWeight','bold');
ylabel('y (m)','fontsize',18,'fontname','Times New Roman','FontWeight','bold');