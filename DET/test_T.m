clc;clear;close all
%% �źŲ�������
select = 2;
Is_noise = 0;
noise = 0;
%% �źŲ���
if(select==1)
    fs = 1024;
    N = 1024+1;
    t = (0:N-1)/fs;
    s1 = cos(2*pi*(200*t));
    s1(t>0.5)=0;
    s2 = cos(2*pi*(200*t)-75*cos(4*pi*t));
    s2(t<=0.5)=0;
    s3 = cos(2*pi*(450*t));
    x = s1+s2+s3;
    %����
    s = 0.01;
else
    N = 2048;
    fs = 200;
    f = (0:N/2)*fs/N;
    t = (0:N-1)/fs;
    A_f1 = exp(0.008*f);
    Phi_f1 = sin(6*pi*f.*f/10000)+3.5*(f);
    GD_t1 = 12*pi*f/10000.*cos(6*pi*f.*f/10000)+3.5;
    X1 = A_f1.*exp(-1i*2*pi*Phi_f1);
    X1(end) = -A_f1(end);
    Y1 = [X1  conj(fliplr(X1(2:end-1)))];    
    s1 = ifft(Y1);   
    x = s1;
    %����
    s = 0.3;
end

if(Is_noise == 1)
    figure
    subplot(2,1,1)
    plot(t,x);
    xlabel('Time (s)','FontSize',20);
    ylabel('Amplitude','FontSize',20);
    title('δ�������Ĵ������ź�');
    set(gca,'FontSize',20);
    x = awgn(x,noise,'measured');
    subplot(2,1,2)
    plot(t,x);
    xlabel('Time (s)','FontSize',20);
    ylabel('Amplitude','FontSize',20);
    title('�������Ĵ������ź�');
    set(gca,'FontSize',20);
else
    figure
    plot(t,x);
    xlabel('Time (s)','FontSize',20);
    ylabel('Amplitude','FontSize',20);
    title('δ�������Ĵ������ź�');
    set(gca,'FontSize',20);
end
 
%% TFM
%�������
stft_type ='T';
tf_type ='TSET2';
direction = 'T';
gamma = 1;
tradeoff = 0.009;
delta = 4;
%����TFR
[Wx,TFx,Ifreq,GD,Rep,Chirprate,q,t,f] = TFM(x,fs,s,stft_type,tf_type,gamma);
%TFR����
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
title(['stft-type:',stft_type,'      tf-type:',tf_type]);
set(gca,'FontSize',20);
%���ӻ���
figure
subplot(1,3,1)
imagesc(t,f,abs(Ifreq'));
axis xy
xlabel('Time (s)','FontSize',20);
ylabel('Frequency (Hz)','FontSize',20);
title('˲ʱƵ��ƽ��');
set(gca,'FontSize',20);
subplot(1,3,2)
imagesc(t,f,abs(GD'));
axis xy
xlabel('Time (s)','FontSize',20);
ylabel('Frequency (Hz)','FontSize',20);
title('Ⱥ�ӳ�ƽ��');
set(gca,'FontSize',20);
subplot(1,3,3)
imagesc(t,f,abs(Chirprate'));
axis xy
xlabel('Time (s)','FontSize',20);
ylabel('Frequency (Hz)','FontSize',20);
title('Chirprateƽ��');
set(gca,'FontSize',20);
%������ȡ
if (strcmp(tf_type, 'STFT')||strcmp(tf_type, 'TSST1')||strcmp(tf_type, 'TSST2')||strcmp(tf_type, 'SST1')||strcmp(tf_type, 'SST2'))
    delta_or_Rep = delta;
else
    delta_or_Rep = Rep;
end
[ExtractTFR,RestTFR] = ExtractOneRidge2SubTFR(TFx,fs,s,direction,tf_type,delta_or_Rep,q,tradeoff);
%��ȡ�������
figure
subplot(2,1,1)
imagesc(t,f,abs(ExtractTFR'));
axis xy
xlabel('Time (s)','FontSize',20);
ylabel('Frequency (Hz)','FontSize',20);
title('��ȡ��TFRƽ��');
set(gca,'FontSize',20);
subplot(2,1,2)
imagesc(t,f,abs(RestTFR'));
axis xy
xlabel('Time (s)','FontSize',20);
ylabel('Frequency (Hz)','FontSize',20);
title('ʣ���TFRƽ��');
set(gca,'FontSize',20);
%�ع�
[Reconstruction,t] = ITFM(ExtractTFR,fs,s,direction,tf_type);
figure
subplot(2,2,1)
plot(t,Reconstruction)
xlabel('Time (s)','FontSize',20);
ylabel('Amplitude','FontSize',20);
title('�ع��ź�');
set(gca,'FontSize',20);
subplot(2,2,2)
plot(t,s1)
xlabel('Time (s)','FontSize',20);
ylabel('Amplitude','FontSize',20);
title('ԭʼ�ź�');
set(gca,'FontSize',20);
subplot(2,1,2)
plot(t,s1)
hold on
plot(t,Reconstruction)
legend('ԭʼ�ź�','�ع��ź�')
xlabel('Time (s)','FontSize',20);
ylabel('Amplitude','FontSize',20);
title('�źŶԱ�');
set(gca,'FontSize',20);