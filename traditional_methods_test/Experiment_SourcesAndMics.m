% ----------- DAS, MUSIC, DAMAS, CLEAN-PSF, CLEAN-SC 波束成像算法
% ------ 可以设置多个不同频率的声源
% ------ 可以添加白噪声
% ------ 可以设置搜索频段

function [X,Y,SPL,Mic_ac] = Experiment_SourcesAndMics(p, Fs, z_source, x_Real, y_Real)
    % 添加路径
    addpath('.\algorithm')
    addpath('.\preprocess')
    addpath('.\algorithm\FISTA_fun')
    addpath('.\data')

    % 麦克风阵列限定区域
    mic_x = [-0.5 0.5];
    mic_y = [-0.5 0.5];

    % 扫描声源限定区域
    scan_x = [-1.5 1.5];
    scan_y = [-1.5 1.5];

    % 绘图显示级数
    BF_dr = 10;
    
    % 待测声压有效值及声速
    c = 343;  

    % 创建麦克风阵列
    createMic;

    % 麦克风坐标 [x,y,z]
    mic_pos = [xcfg ycfg];
    mic_pos(:,3) = 0;

    % 分辨率步长
    step = 0.1;

    % 添加白噪声
    % addWhiteNoise;
    
    % 确定扫描频段（2k-52k）
    search_freql = 800;  
    search_frequ = 4000;

    % 计算CSM以及确定扫描频率
    [CSM, freqs] = developCSM(p.', search_freql, search_frequ, Fs);

    % 计算steering vector
    h = steerVector(z_source, freqs, [scan_x scan_y], step, mic_pos.', c);

    % 波束成像
%     % DAS算法
%       [X, Y, B] = DAS(CSM, h, freqs, [scan_x scan_y], step);

%    % Test
%     	[X, Y, B] = Test(CSM, h, freqs, [scan_x scan_y], step, flag);
%
%     % MUSIC算法
%     [X, Y, B] = MUSIC(CSM, h, freqs, [scan_x scan_y], step, mic_pos.');
%
%     % DAMAS算法
    [X, Y, B] = DAMAS(CSM, h, freqs, [scan_x scan_y], step, mic_pos.');
%
%     % CLEAN-PSF算法
%     [X, Y, B] = CleanPSF(CSM, h, z_source, freqs, [scan_x scan_y], step, mic_pos.');
%
%     % CLEAN-SC算法
%     [X, Y, B] = CleanSC(CSM, h, z_source, freqs, [scan_x scan_y], step);
%     
%     % 创建 PSF
%       PSF = createPSF(z_source, freqs, [scan_x scan_y], step, mic_pos.', c);

%     % FISTA算法
%     [X, Y, B] = FISTA(PSF, CSM, h, freqs, [scan_x scan_y], step);
%
%     % FFT-NNLS算法
%     [X, Y, B] = FFTNNLS(PSF, CSM, h, freqs, [scan_x scan_y], step, flag);
      
%     % FFT-ADMM算法
%       [X, Y, B] = FFTADMM(PSF, CSM, h, freqs, [scan_x scan_y], step, flag);
     
%      % FFT-DFISTA算法
%      [X, Y, B] = DFISTA(PSF, CSM, h, freqs, [scan_x scan_y], step, flag);
    
    % 声压级单位转换
    B(B<0)=0;
    SPL = 20*log10(eps+sqrt(real(B))/2e-5);
    SPL(SPL<0)=0;
%     SPL = SPL-max(max(SPL));
    
    % 绘制功率图
    plotBeam;

    % figure;
    % contourf(X, Y, SPL, [(maxSPL-BF_dr):1:maxSPL], 'Parent', BeamformFigure);
    % contourf(X, Y, SPL);
end