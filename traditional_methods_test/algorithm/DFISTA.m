function [X, Y, B_sum] = DFISTA(PSF, CSM, h, frequencies,...
    scan_limits, grid_resolution, maxIter)

D = zeros(2*size(PSF(:,:,1)));
for i = size(D,1)-1
    D(i,i) = 1; D(i,i+1) = -1;
end
D(size(D,1),size(D,1)) = 1;
D = sparse(D);

fprintf('\t------------------------------------------\n');
fprintf('\tStart beamforming, DFISTA...\n');

% Setup scanning grid using grid_resolution and dimensions
N_freqs = length(frequencies);

X = scan_limits(1):grid_resolution:scan_limits(2);
Y = scan_limits(3):grid_resolution:scan_limits(4);
N_X = length(X);
N_Y = length(Y);

if ~exist('maxIter', 'var')
    maxIter = 50;
end

lambda = 10;

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

    % Compute Lipschitz constant
    L = lipschitz1(PSF_K,Fps,D,lambda);

    % For n = 1
    [~,grady] = gradient(Fps,B,y,D,lambda);
    
    % Start iteration
    n = 0;
    while n < maxIter 
        n = n+1;
        x = max(0,y - (1/L)*grady);

        tnew = (1+sqrt(1+4*t*t))/2;
        y = x + ((t-1)/tnew)*(x-xold);

        [~,grady] = gradient(Fps,B,y,D,lambda);
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

function L = lipschitz1(PSF,Fps,D,lambda)
% Estimate Lipschitz constant by power iteration
x = rand(size(PSF));
for k = 1:10
    x = (fftshift(ifft2(fft2(x).*Fps))+lambda*D*x)/norm(x,'fro');
end
    L = norm(x,'fro')^2;    % lambda(A'A) Assuming a symmetric matric A
end

function [f,g,r] = gradient(Fps,b,x,D,lambda)
    r = fftshift(ifft2(fft2(x).*Fps)) - b;
    f = 0.5*norm(r,'fro')^2;
    if (nargout > 1)
        g = fftshift(ifft2(fft2(r).*Fps))+lambda*D*x;
    end
end
