clear all;clc;
% addpath('utils/')
count = 0;
SampFreq = 51200;
stft_type ='T';
tf_type ='STFT';
direction = 'T';
gamma = 1;
tradeoff = 0.009;
delta = 4;
s = 1e-4;
source_path = '/home2/zgx/data/data_dir/';
save_path = '/home2/zgx/data/single_sound_source_stft/';
source_dir = dir(source_path);

for i=3:length(source_dir)
    data = h5read([source_path,'/', source_dir(i).name],'/time_data');
%     mkdir(save_path, source_dir(i).name);
    for j=1:1
    Sig = data(j,1:1024);
    [~,TFx,~,~,Rep,~,q,t,f] = TFM(Sig,SampFreq,s,stft_type,tf_type,gamma);
    STFT_TFD = abs(TFx');
%     Gray_STFT_TFD = mat2gray(STFT_TFD);
%    
%     save_dirname = [save_path, source_dir(i).name];
%     imwrite(Gray_STFT_TFD, [save_dirname, '/',num2str(j),'.png']);
    end
    figure
    imagesc(t,f,STFT_TFD);
%     figure
%     imshow(Gray_STFT_TFD);
    count = count + 1
end






% save 1.mat STFT_TFD
