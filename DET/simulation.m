close all;clc; clear all;
%% 导入信号
fs = 1024;
N =1024;
t = (0:N-1)/fs;
% a1
a1 = 1;
phi1 = 330*t+16*cos(3*pi*t);
if1 = 330-48*pi*sin(3*pi*t); 
s1 = a1.*cos(2*pi*(phi1));
% a2
a2 = exp(-0.6*t);
phi2 = 190*t+9*cos(3*pi*t);
if2 = 190-27*pi*sin(3*pi*t); 
s2 = a2.*cos(2*pi*(phi2));
% a3
a3=2*exp(-8*(t-0.5).^2);
a3=1;
phi3 = 40*t;
if3 = (ones(length(t),1)*40)'; 
s3 = a3.*cos(2*pi*(phi3));
figure;
plot(t,if1,t,if2,t,if3);

x = s1+s2+s3;
figure;
plot(t,x);title('s1+s2+s3');
% x = awgn(x,20,'measured');
%% 参数
s = 0.01;
tradeoff = 0.009;
gamma = 1;
%% TFR
[Wx,TFx,Rep_t,Rep_m,q_t,q_m,t,f] = DET(x,fs,s,gamma);

figure
subplot(2,1,1)
imagesc(t,f,abs(Wx'));
axis xy
xlabel('Time (s)','FontSize',20);
ylabel('Frequency (Hz)','FontSize',20);
title('STFT');
set(gca,'FontSize',20);
subplot(2,1,2)
imagesc(t,f,abs(TFx'));
axis xy
xlabel('Time (s)','FontSize',20);
ylabel('Frequency (Hz)','FontSize',20);
title('TFx');
set(gca,'FontSize',20);
%% 提取
[ExtractTFR1,RestTFR1] = ExtractOneRidge2SubTFR(TFx,fs,s,'F','DET',Rep_m,q_m,tradeoff);
[ExtractTFR2,RestTFR2] = ExtractOneRidge2SubTFR(RestTFR1,fs,s,'F','DET',Rep_m,q_m,tradeoff);
[ExtractTFR3,RestTFR3] = ExtractOneRidge2SubTFR(RestTFR2,fs,s,'F','DET',Rep_m,q_m,tradeoff);

figure
imagesc(t,f,abs(ExtractTFR1'));
axis xy
xlabel('Time (s)','FontSize',20);
ylabel('Frequency (Hz)','FontSize',20);
title('ExtractTFR1');
set(gca,'FontSize',20);

figure
imagesc(t,f,abs(ExtractTFR2'));
axis xy
xlabel('Time (s)','FontSize',20);
ylabel('Frequency (Hz)','FontSize',20);
title('ExtractTFR2');
set(gca,'FontSize',20);

figure
imagesc(t,f,abs(ExtractTFR3'));
axis xy
xlabel('Time (s)','FontSize',20);
ylabel('Frequency (Hz)','FontSize',20);
title('ExtractTFR3');
set(gca,'FontSize',20);
%% 重构
[ExtractTFR1] = ConvertSTFT(ExtractTFR1,t,f,'MSTFT',0.001);
[Reconstruction1,t] = ITFM(ExtractTFR1,fs,s,'F','DET');
figure
plot(t,Reconstruction1,t,s3)
xlabel('Time (s)','FontSize',20);
ylabel('Amplitude','FontSize',20);
title('s3');
set(gca,'FontSize',20);

[ExtractTFR2] = ConvertSTFT(ExtractTFR2,t,f,'MSTFT',0.001);
[Reconstruction2,t] = ITFM(ExtractTFR2,fs,s,'F','DET');
figure
plot(t,Reconstruction2,t,s1)
xlabel('Time (s)','FontSize',20);
ylabel('Amplitude','FontSize',20);
title('s1');
set(gca,'FontSize',20);

[ExtractTFR3] = ConvertSTFT(ExtractTFR3,t,f,'MSTFT',0.001);
[Reconstruction3,t] = ITFM(ExtractTFR3,fs,s,'F','DET');
figure
plot(t,Reconstruction3,t,s2)
xlabel('Time (s)','FontSize',20);
ylabel('Amplitude','FontSize',20);
title('s2');
set(gca,'FontSize',20);