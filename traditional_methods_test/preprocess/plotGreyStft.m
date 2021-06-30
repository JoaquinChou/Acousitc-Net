clear all;clc;
count = 0;
SampFreq = 51200;
stft_type ='T';
tf_type ='STFT';
direction = 'T';
gamma = 1;
tradeoff = 0.009;
delta = 4;
s = 1e-4;
source_path = 'E:/研究生/实验室工作/涂老师/声学会议/traditional_methods_test/data';
save_path = '/home2/zgx/data/single_sound_source_stft/';
source_dir = dir(source_path);

for i = 1:1
    data = h5read([source_path,'/', source_dir(i+2).name],'/time_data');
    for j=1:1
    %         画出时域信号图像
        plot(data(j,:));
    %   axis off;
        xlim([0 1024]);
        title(['Original Wave'],'fontsize',20);
        ylabel('amplitude ','fontsize',18);
        xlabel('time (s)','fontsize',18);
        Sig = data(j,1:1024);   
        [~,TFx,~,~,Rep,~,q,t,f] = TFM(Sig,SampFreq,s,stft_type,tf_type,gamma);
        STFT_TFD = abs(TFx');
        Gray_STFT_TFD = mat2gray(STFT_TFD);
        

    end

    
    
    figure
    imagesc(t, f, STFT_TFD);
    title(['STFT Result'],'fontsize',20);
    ylabel('Frequency (Hz)','fontsize',18)
    xlabel('time (s)','fontsize',18)
    figure
    imagesc(t, f, Gray_STFT_TFD * 255);
    colormap(gray(256));
    title(['Grey Image'],'fontsize',20);
    ylabel('Frequency (Hz)','fontsize',18)
    xlabel('time (s)','fontsize',18)
 end