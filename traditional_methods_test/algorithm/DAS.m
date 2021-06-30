function [X, Y, B] = DAS(CSM, h, frequencies, scan_limits, grid_resolution)


% 设置扫描网格
N_freqs = length(frequencies);

X = scan_limits(1):grid_resolution:scan_limits(2);
Y = scan_limits(3):grid_resolution:scan_limits(4);
N_X = length(X);
N_Y = length(Y);

B = zeros(1,N_X*N_Y);
for K = 1:N_freqs  
    
    % 累加不同频段的信号
    B = B + sum(h(:,:,K).*(CSM(:,:,K)*conj(h(:,:,K))), 1); 
    
end

B = reshape(B, N_X, N_Y).';

end