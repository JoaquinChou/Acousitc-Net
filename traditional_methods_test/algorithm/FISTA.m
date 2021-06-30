function [X, Y, B_sum] = FISTA(PSF, CSM, h, frequencies,...
    scan_limits, grid_resolution, maxIter)

fprintf('\t------------------------------------------\n');
fprintf('\tStart beamforming, FISTA...\n');

% Setup scanning grid using grid_resolution and dimensions
N_freqs = length(frequencies);

X = scan_limits(1):grid_resolution:scan_limits(2);
Y = scan_limits(3):grid_resolution:scan_limits(4);
N_X = length(X);
N_Y = length(Y);

if ~exist('maxIter', 'var')
    maxIter = 50;
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

    % FISTA:
    % Initialize variables
    x = x0;
    xold = x;
    y = x;
    t = 1; 

    % Precompute fft of PSF
    Fps = fft2(PSF_K);
    FpsT = fft2(rot90(PSF_K,2));

    % Evaluate f(x0)
    fgx = @(x) nnls(PSF_K,B,x,Fps,FpsT);      

    % Compute Lipschitz constant
    L = lipschitz(PSF_K,Fps);

    % For n = 1
    [~,grady] = fgx(y);
    
    % Start iteration
    n = 0;
    while n < maxIter    

        n = n+1;
        x = max(0,y - (1/L)*grady);

        tnew = (1+sqrt(1+4*t*t))/2;
        y = x + ((t-1)/tnew)*(x-xold);

        [~,grady] = fgx(y);
        xold = x;
        t = tnew;  
    end
    %-------end FISTA
    
    % Remove zero-padding
    x = x(int64(N_X/2)+1:int64(N_X/2 + N_X),int64(N_X/2)+1:int64(N_X/2 + N_X));

    B_sum = B_sum + x;

end

fprintf('\tBeamforming complete!\n');
fprintf('\t------------------------------------------\n');  
end
