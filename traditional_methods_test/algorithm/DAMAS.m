function [X,Y,B_sum] = DAMAS(CSM, h, frequencies,...
    scan_limits, grid_resolution, mic_positions, maxIter)

fprintf('\t------------------------------------------\n');
fprintf('\tStart beamforming, DAMAS...\n');

% Setup scanning grid using grid_resolution and dimensions
N_mic = size(mic_positions, 2);
N_freqs = length(frequencies);

X = scan_limits(1):grid_resolution:scan_limits(2);
Y = scan_limits(3):grid_resolution:scan_limits(4);
N_X = length(X);
N_Y = length(Y);

if ~exist('maxIter', 'var')
    maxIter = 1000;
end

fprintf('\tBeamforming for %d frequency points...\n', N_freqs);

B_sum = zeros(N_X, N_Y);deps = 1e-3;
for K = 1:N_freqs
    P = sum(h(:,:,K).*(CSM(:,:,K)*conj(h(:,:,K))), 1);
    P = reshape(P, N_X, N_Y).';

    %Initialise final source powers B
    temp_B = real(P);
    B = zeros(size(temp_B));
    B0 = temp_B;

    %Solve the system Y = AQ for Q by Gauss-Seidel iteration where Y is the
    %original delay-and-sum plot we want to deconvolve, and Q are the true
    %source powers
    A = (abs(h(:,:,K)'*h(:,:,K)).^2)./N_mic^2;
    for i = 1:maxIter

        %Gauss-Seidel iteration. If the solution is negative set it to zero (to
        %ensure that we only have positive and not negative power)
        for n = 1:N_X*N_Y
            B(n) = max(0, temp_B(n) - A(n, 1:n-1)*B(1:n-1)' ...
                - A(n, n+1:end)*B0(n+1:end)');
        end

        %Break criterion for convergence
        dX = (B - B0);
        maxd = max(abs(dX(:)))/mean(B0(:));

        if  maxd < deps
            break;
        end

        B0 = B;
    end
    
    B_sum = B_sum + B;
end
    
if i == maxIter
    disp(['Stopped after maximum iterations (' num2str(maxIterations) ')'])
else
    disp(['Converged after ' num2str(i) ' iterations'])
end

end
