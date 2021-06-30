function L = lipschitz(PSF,Fps)
% Estimate Lipschitz constant by power iteration
x = rand(size(PSF));
for k = 1:10
    x = fftshift(ifft2(fft2(x).*Fps))/norm(x,'fro');
end
    L = norm(x,'fro')^2;    % lambda(A'A) Assuming a symmetric matric A
end
