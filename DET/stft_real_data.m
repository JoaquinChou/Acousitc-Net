
load('C:\Users\Joaquin Chou\Desktop\研一下\Beamform\DET\x_0_y_0_rms_70.6_sources.mat');
save_dirname = 'C:\Users\Joaquin Chou\Desktop\x_0_y_0_rms_70.6_operasources.mat';
count = 0;
for i=1:56
    trum_data = data(1:1024,i);

    SampFreq = 51200;
    Sig = trum_data;

    stft_type ='T';
    tf_type ='STFT';
    direction = 'T';
    gamma = 1;
    tradeoff = 0.009;
    delta = 4;

    s = 1e-3;
    [~,TFx,~,~,Rep,~,q,t,f] = TFM(Sig,SampFreq,s,stft_type,tf_type,gamma);
    STFT_TFD = abs(TFx');
    Gray_STFT_TFD = mat2gray(STFT_TFD);
    
    imwrite(Gray_STFT_TFD,[save_dirname,'\',num2str(i),'.png']);
    
    count = count + 1
        
%     figure
%     imagesc(t,f,Gray_STFT_TFD);
end