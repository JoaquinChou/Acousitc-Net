function [X, Y, B_sum] = FFTNNLS(PSF, CSM, h, frequencies,...
    scan_limits, grid_resolution, maxIter)

fprintf('\t------------------------------------------\n');
fprintf('\tStart beamforming, FFT-NNLS...\n');

% Setup scanning grid using grid_resolution and dimensions
N_freqs = length(frequencies);

X = scan_limits(1):grid_resolution:scan_limits(2);
Y = scan_limits(3):grid_resolution:scan_limits(4);
N_X = length(X);
N_Y = length(Y);

if ~exist('maxIter', 'var')
    maxIter = 100;
end

fprintf('\tBeamforming for %d frequency points...\n', N_freqs);

B_sum = zeros(N_X, N_Y);

for K = 1:N_freqs
    % DAS结果
    B = sum(h(:,:,K).*(CSM(:,:,K)*conj(h(:,:,K))), 1);
    B = reshape(B, N_X, N_Y).';
    
    % zero-padding
    PSF_K = PSF(:,:,K);
    B = real(zeropad(B));
    PSF_K = zeropad(PSF_K);
    x0 = zeros(2*N_X);
    
    % FFT-NNLS
    x = x0;

    % Precompute fft of PSF
    Fps = fft2(PSF_K);
    FpsT = fft2(rot90(PSF_K,2));
    
    fgx = @(x) nnls(PSF_K,B,x,Fps,FpsT); 

    % 开始迭代
    n = 0;
    while n < maxIter
        n = n+1;
        [~,grad,r] = fgx(x);        
        d = grad;                   
        d(x == 0 & d > 0) = 0;      

        g = fftshift(ifft2(fft2(d).*Fps));         
        t = dot(g(:),r(:))/dot(g(:),g(:));    

        x = max(0,x - t*d);  
    end
    % ------------end FFT-NNLS
    
    % Remove zero-padding
    x = x(int64(N_X/2)+1:int64(N_X/2 + N_X),int64(N_X/2)+1:int64(N_X/2 + N_X));
    
    B_sum = B_sum + x;
end

fprintf('\tBeamforming complete!\n');
fprintf('\t------------------------------------------\n');  

end