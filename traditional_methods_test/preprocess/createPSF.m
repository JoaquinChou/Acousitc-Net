function PSF = createPSF(plane_distance, frequencies, ...
    scan_limits, grid_resolution, mic_positions, c)
% ------ 计算点扩散函数

fprintf('\t------------------------------------------\n');
fprintf('\tStart calculating PSF...\n');

% Setup scanning grid using grid_resolution and dimensions
N_mic = size(mic_positions, 2);
N_freqs = length(frequencies);

X = scan_limits(1):grid_resolution:scan_limits(2);
Y = scan_limits(3):grid_resolution:scan_limits(4);
Z = plane_distance;
N_X = length(X);
N_Y = length(Y);

% 初始化变量
PSF = zeros(N_X, N_Y, N_freqs);
dj = zeros(N_X, N_Y, N_mic);
ej = zeros(N_X, N_Y, N_mic);
g_rs = zeros(N_mic,1);

d0 = sqrt(repmat(X,N_Y,1).^2 + repmat(Y',1,N_X).^2 + Z^2);  
for K = 1:N_freqs
    for m = 1:N_mic
        k = 2*pi*frequencies(K)/c;

        % 扫描平面到第m个麦克风距离
        dj(:,:,m) = sqrt((repmat(X,N_Y,1)-mic_positions(1,m)).^2 + ...
            (repmat(Y',1,N_X)-mic_positions(2,m)).^2 + Z^2);

        % v(r)
        ej(:,:,m) = (dj(:,:,m)./d0).*exp(1j*k.*dj(:,:,m));

        % g(rs): 扫描平面中心点
        d_rs = sqrt((0-mic_positions(1,m)).^2 + ...
            (0-mic_positions(2,m)).^2 + Z^2);
        g_rs(m) = (Z./d_rs).*exp(1j*k*d_rs);
    end

    % 计算PSF
    for ii = 1:N_X
        for jj = 1:N_Y
            PSF(ii,jj, K) = dot(squeeze(ej(ii,jj,:)),g_rs);
        end
    end
    
    PSF(:,:,K) = rot90(abs(PSF(:,:,K)).^2/N_mic^2);      % Normalize PSF
end


fprintf('\tFinished calculating PSF!\n');
fprintf('\t------------------------------------------\n');
end