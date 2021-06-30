function [X, Y, B] = CleanSC(CSM, h, plane_distance, frequencies, ...
    scan_limits, grid_resolution)
% Beamforming using CLEAN SC according to Sijtsma 2007 CLEAN paper.
% Using steering vector formulation III 'by' Ennes Sarradj. I.e. it 
% is the same as SarradjBeamforming.m using form3. No diagonal removal.
% Variables of interest are the:
% - Clean beam width, lambda
% - Loop gain, phi
% - Stop criteria, epsilon
%
% Anwar Malgoezar, Oct. 2017
% Group ANCE

fprintf('\t------------------------------------------\n');
fprintf('\tStart beamforming, CLEAN-SC...\n');

lambda = 480;
phi = 0.99; % loop gain

% Setup scanning grid using grid_resolution and dimensions
N_freqs = length(frequencies);

X = scan_limits(1):grid_resolution:scan_limits(2);
Y = scan_limits(3):grid_resolution:scan_limits(4);
Z = plane_distance;
N_X = length(X);
N_Y = length(Y);
N_Z = length(Z);

N_scanpoints = N_X*N_Y*N_Z;
x_t = zeros(N_scanpoints, 3);

x_t(:, 1) = repmat(X, 1, N_Y);
dummy = repmat(Y, N_X, 1);
x_t(:, 2) = dummy(:);
x_t(:, 3) = plane_distance;

B = zeros(1, N_scanpoints);
B_mix = zeros(N_freqs, N_scanpoints);

fprintf('\tBeamforming for %d frequency points...\n', N_freqs);
for K = 1:N_freqs
    Q = zeros(1, N_scanpoints);
   
    % Start CLEAN SC procedure
    % Eq. (9), P^(0)
    
    hk = h(:,:,K);
    D = conj(CSM(:,:,K)); % To stick to Pieter's conjugate matrix multiplication, I take here the conjugate of the CSM
    P = sum(conj(hk).*(D*hk), 1);
    
    % Start iteration
    Cabs = sum(abs(D(:)));
    Dcurr = Cabs;
    count = 0;

    Dprev = 1e10;
    while ( Dcurr < Dprev )&&(count<10)
        % Determine peak source
        [Pmax, index_max] = max(abs(P));
        ispositiv = real(P(index_max)) > 0; % Do not 'include' negative pressure maps
        
        gmax = D*hk(:,index_max)/Pmax;
        % Degraded CSM Eq. (14)
        Cource = Pmax*(gmax)*gmax';
        D = D - phi*Cource;
        % Degraded source power Eq. (15), P^(i)
        P = sum(conj(hk).*(D*hk), 1);
        
        % Construct clean beam Eq. (13)
        d_frompeak = sqrt( (x_t(:,1) - x_t(index_max,1)).^2 + ...
                           (x_t(:,2) - x_t(index_max,2)).^2 );
        Q = Q + phi*Pmax*10.^(-lambda.*d_frompeak.^2).' * ispositiv;
        
        Dprev = Dcurr;
        Dcurr = sum(abs(D(:)));
        count = count + 1;

    end
    fprintf('\tf = %.2f, # of iterations: %d\n', frequencies(K), count);
    
    % Remaining dirty map shouldnt contain negative pressures
    P(real(P)<0) = 0;
    % Add to final map, i.e. sum over frequencies, clean map and residual
    % dirty map
    B = B + Q + P;
    B_mix(K,:) = B;
end

B_mix = sum(B_mix,1);
B = reshape(B_mix, N_X, N_Y).';

fprintf('\tCLEAN-SC procedure completed!\n');
fprintf('\t------------------------------------------\n');
end