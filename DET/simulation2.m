close all;clc; clear all;
%% 导入信号
%% Parameters
    N = 2048;
    fs = 200;
    f = (0:N/2)*fs/N;
%% Test Signal    
%Mode1
    A_f1 = exp(0.008*f);
    Phi_f1 = sin(6*pi*f.*f/10000)+3.5*(f);
    GD_t1 = 12*pi*f/10000.*cos(6*pi*f.*f/10000)+3.5;
    X1 = A_f1.*exp(-1i*2*pi*Phi_f1);
    X1(end) = -A_f1(end);
    Y1 = [X1  conj(fliplr(X1(2:end-1)))];    
    y1 = ifft(Y1);
%Mode2
    A_f2 = exp(0.005*f);
    Phi_f2 = 6*f+0.07/2*f.^2-0.0007/3*f.^3;
    GD_t2 = 6+0.07*f-0.0007*f.^2;
    X2 = A_f1.*exp(-1i*2*pi*Phi_f2);
    X2(end) = -A_f1(end);
    Y2 = [X2  conj(fliplr(X2(2:end-1)))];    
    y2 = ifft(Y2);
%Signal
    x = y1+y2;
%     x = awgn(x,20,'measured');
%% 参数
s = 0.3;
tradeoff = 1.2;
gamma = 0.5;
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
[ExtractTFR1,RestTFR1] = ExtractOneRidge2SubTFR(TFx,fs,s,'T','DET',Rep_t,q_t,tradeoff);
[ExtractTFR2,RestTFR2] = ExtractOneRidge2SubTFR(RestTFR1,fs,s,'T','DET',Rep_t,q_t,tradeoff);

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
%% 重构
[Reconstruction1,t] = ITFM(ExtractTFR1,fs,s,'T','DET');
figure
plot(t,Reconstruction1)
xlabel('Time (s)','FontSize',20);
ylabel('Amplitude','FontSize',20);
title('y2');
set(gca,'FontSize',20);

[Reconstruction2,t] = ITFM(ExtractTFR2,fs,s,'T','DET');
figure
plot(t,Reconstruction2)
xlabel('Time (s)','FontSize',20);
ylabel('Amplitude','FontSize',20);
title('y1');
set(gca,'FontSize',20);
