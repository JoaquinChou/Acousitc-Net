function [X, Y, B_sum] = FFTADMM(PSF, CSM, h, frequencies, scan_limits, grid_resolution, maxIter, lambda, rho)
%   minimize 1/2*|| F-1[F(x).*F(PSF)]-b]||_F^2 +lambda*||Dx||_1
%   s.t. x>=0
%
% The solution is returned in the vector x.

D = zeros(2*size(PSF(:,:,1)));
for i = size(D,1)-1
    D(i,i) = 1; D(i,i+1) = -1;
end
D(size(D,1),size(D,1)) = 1;
D = sparse(D);

fprintf('\t------------------------------------------\n');
fprintf('\tStart beamforming, FFT-ADMM...\n');

if ~exist('lambda', 'var')
    lambda = 10;
end

if ~exist('rho', 'var')
    rho = 1;
end

if ~exist('maxIter', 'var')
    maxIter = 50;
end

% Setup scanning grid using grid_resolution and dimensions
N_freqs = length(frequencies);

X = scan_limits(1):grid_resolution:scan_limits(2);
Y = scan_limits(3):grid_resolution:scan_limits(4);
N_X = length(X);
N_Y = length(Y);

fprintf('\tBeamforming for %d frequency points...\n', N_freqs);

B_sum = zeros(N_X, N_Y);
for K = 1:N_freqs
    K
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
    % ADMM solver
    x = x0;
    z = x;
    u = x;

    for k = 1:maxIter

        % x-update
        x = FISTA_solver(x,PSF_K,B,z,D,rho,u);

        % z-update 
        x_hat = D*x;
        z = shrinkage(D*x + u, lambda/rho);

        % u-update
        u = u + (x_hat - z);

    end

    % Remove zero-padding
    x = x(int64(N_X/2)+1:int64(N_X/2 + N_X),int64(N_X/2)+1:int64(N_X/2 + N_X));

    B_sum = B_sum + x;

end

fprintf('\tBeamforming complete!\n');
fprintf('\t------------------------------------------\n');  
end


% function z = shrinkage(x, kappa)
% temp = abs(x);
% if temp < kappa
%     z = zeros(size(x));
% else
%     z = [temp>kappa].*x.* (temp-kappa)./temp;
% end
% end
function z = shrinkage(x, kappa)
    z = max( 0, x - kappa ) - max( 0, -x - kappa );
end

