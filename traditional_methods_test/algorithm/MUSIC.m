function [X,Y,B_sum,B] = MUSIC(CSM, h, frequencies, ...
    scan_limits, grid_resolution, mic_positions, nSources)
%steeredResponseDelayAndSum - calculate delay and sum in frequency domain
%
%Calculates the steered response from the delay-and-sum algorithm in the
%frequency domain based on sensor positions, input signal and scanning angles
%
%S = steeredResponseDelayAndSum(R, e, w)
%
%IN
%R - PxP correlation matrix / cross spectral matrix (CSM)
%e - MxNxP steering vector/matrix for a certain frequency
%w - 1xP vector of element weights
%
%OUT
%S - MxN matrix of delay-and-sum steered response power
%
%Created by J?rgen Grythe, Squarehead Technology AS
%Last updated 2017-01-31

N_mic = size(mic_positions, 2);
N_freqs = length(frequencies);

X = scan_limits(1):grid_resolution:scan_limits(2);
Y = scan_limits(3):grid_resolution:scan_limits(4);
N_X = length(X);
N_Y = length(Y);

if ~exist('nSources', 'var')
    nSources = 1;
end

fprintf('\tBeamforming for %d frequency points...\n', N_freqs);
B = zeros(N_X, N_Y, N_freqs);
for K = 1:N_freqs

    hk = h(:,:,K);
    CSM_temp = CSM(:,:,K);
   
    % Cross spectral matrix with diagonal loading
    CSM_temp = CSM_temp + trace(CSM_temp)/(N_mic^2)*eye(N_mic, N_mic);
    CSM_temp = CSM_temp/N_mic;

    % Eigenvectors of R
    [Vec, Val]=eig(CSM_temp);                           
    [~, Seq]=sort(max(Val));
    
    % Noise eigenvectors
    Vn = Vec(:,Seq(1:end-nSources));    

    e = hk'; i =1;
    B0 = zeros(N_X, N_Y); 
    
    if abs(max(sum(Val,1)))<1
        B(:,:,K) = B0;
        continue
    end
    
     if abs(max(sum(Val,1)))>1
        disp([abs(max(sum(Val,1))) frequencies(K)])
     end
    
    for x = 1:N_X 
        for y = 1:N_Y    
            ee = reshape(e(i,:), N_mic, 1); i=i+1;
            B0(x, y) = 1./(ee'*(Vn*Vn')*ee);
        end
    end
       
%     if abs(max(sum(Val,1)))<1
%         B0 = zeros(N_X, N_Y);
%     end
    
    B(:,:,K) = B0;
end

% 累加各个频段的功率谱图
B_sum = zeros(N_X, N_Y);
for K = 1:N_freqs
    B_sum = B_sum + reshape(B(:,:,K), N_X, N_Y);
end


end