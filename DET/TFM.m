function [Wx,TFx,Ifreq,GD,Rep,Chirprate,q,t,f] = TFM(x,fs,s,stft_type,tf_type,gamma)
%% 参数说明
%函数名：时频分析方法；英文名：Time-Frequency Method（TFM）
%%%%%%%%%%%%%%输入参数%%%%%%%%%%%%%%
%x:信号向量
%fs:采样率
%s:窗长度
%stft_type:所用短时傅里叶变换的类型
%tf_type:时频变换类型
%gamma:阈值
%%%%%%%%%%%%%%输出参数%%%%%%%%%%%%%%
%Wx：短时傅里叶的TFR
%TFx：改进重排变换的TFR
%Ifreq、GD：瞬时频率算子以及群延迟算子
%Chirprate：Chirprate算子
%Rep、q：重构因子
%t:时间序列
%f:频率序列
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%所支持的时频方法：
%时间方向：
%STFT :短时傅里叶变换
%RM   :重排变换
%TSST1:一阶时间同步压缩变换；
%TSST2:二阶时间同步压缩变换；
%TSET1:一阶时间同步提取变换；
%TSST2:二阶时间同步提取变换；
%MRM  :改进重排变换；
%SST1 :一阶频率同步压缩变换(仅用于对比)；
%频率方向：
%STFT :短时傅里叶变换
%RM   :重排变换
%SST1:一阶频率同步压缩变换；
%SST2:二阶频率同步压缩变换；
%SET1:一阶频率同步提取变换；
%SST2:二阶频率同步提取变换；
%MRM   :改进重排变换；
%TSST1 :一阶时间同步压缩变换(仅用于对比)；
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 参数数目检查
    if (nargin > 6)
        error('输入参数过多！');
    elseif(nargin == 5)
        gamma = 0;  
    elseif(nargin == 4)
        gamma = 0; tf_type = 'STFT';
    elseif(nargin == 3)
        gamma = 0; tf_type = 'STFT';stft_type = 'T';
    elseif(nargin == 2 || nargin == 1 || nargin == 0 )
        error('缺少输入参数！');
    end
%stft_type检查
    if ((~strcmp(stft_type, 'Traditional')) && (~strcmp(stft_type, 'T')) && ...
        (~strcmp(stft_type, 'Modified')) && (~strcmp(stft_type, 'M')))
        error(['stft_type不支持：',stft_type,'类型，',...
            '本函数仅支持：Traditional、T、Modified、M']);
    end
%tf_type检查
    if ((~strcmp(tf_type, 'STFT')) && (~strcmp(tf_type, 'RM')) && ...
        (~strcmp(tf_type, 'SST1')) && (~strcmp(tf_type, 'SST2')) && ...
        (~strcmp(tf_type, 'SET1')) && (~strcmp(tf_type, 'SET2')) && ...
        (~strcmp(tf_type, 'TSST1')) && (~strcmp(tf_type, 'TSST2')) && ...
        (~strcmp(tf_type, 'TSET1')) && (~strcmp(tf_type, 'TSET2')) && (~strcmp(tf_type, 'MRM')))
        error(['tf_type不支持：',tf_type,'类型，',...
            '本函数仅支持：STFT、RM、SST1、SST2、SET1、SET2、TSST1、TSST2、TSET1、TSET2、MRM']);
    end
%输出赋值
    Wx=0;TFx=0;Ifreq=0;GD=0;q=0;Rep=0;Chirprate=0;t=0;f=0;
%% 预处理
%判断是否为列向量，是则转置
    [xrow,~] = size(x);
    if (xrow~=1)
        x = x';
    end
%% Traditional时频计算
    if (strcmp(stft_type, 'Traditional') || strcmp(stft_type, 'T'))
%%%%%%%%%%%%%计算公共部分%%%%%%%%%%%%%%%%
        N = length(x);
        t = (0:N-1)/fs;
        tao = (0:N-1)/fs;
        if mod(N,2)==0
            L = N/2+1;
        else
            L = (N+1)/2;
        end
        f = fs/N*(0:L-1);
        TFx = zeros(N,L);
        dt = 1/fs;
        df = f(2)-f(1);
%%%%%%%%%%%%%计算STFT%%%%%%%%%%%%%%%%
        if (strcmp(tf_type, 'STFT'))
            Wx = zeros(N,L);
            gt = @(t) s^(-1/2)*pi^(-1/4).*exp(-t.^2/s^2/2);
            for ptr = 1:N
                gh = gt(t-tao(ptr));
                gh = conj(gh);
                xcpsi = fft(gh .* x);
                Wx(ptr,:) = xcpsi(1:L);
            end
            TFx = Wx;
%%%%%%%%%%%%%计算RM%%%%%%%%%%%%%%%%
        elseif (strcmp(tf_type, 'RM'))
            %G(w)
            gt = @(t) s^(-1/2)*pi^(-1/4).*exp(-t.^2/s^2/2);
            Wx = zeros(N,L);
            for ptr = 1:N
                gh = gt(t-tao(ptr));
                gh = conj(gh);
                xcpsi = fft(gh .* x);
                Wx(ptr,:) = xcpsi(1:L);
            end
            %DG(w)
            gt = @(t) (-1j*t)*s^(-1/2)*pi^(-1/4).*exp(-t.^2/s^2/2);
            dWx = zeros(N,L);
            for ptr = 1:N
                gh = gt(t-tao(ptr));
                gh = conj(gh);
                xcpsi = fft(gh .* x);
                dWx(ptr,:) = xcpsi(1:L);
            end
            %wG(w)
            gt = @(t) (1j*t)/s/s*s^(-1/2)*pi^(-1/4).*exp(-t.^2/s^2/2);
            wWx = zeros(N,L);
            for ptr = 1:N
                gh = gt(t-tao(ptr));
                gh = conj(gh);
                xcpsi = fft(gh .* x);
                wWx(ptr,:) = xcpsi(1:L);
            end
            %计算瞬时频率
            Ifreq = zeros(N,L);
            for k = 1:N
                for m = 1:L
                    Ifreq(k,m) = 2*pi*f(m)+conj(wWx(k,m)./Wx(k,m));
                end
            end
            Ifreq = Ifreq/2/pi;
            Ifreq(abs(Wx)<gamma)=0;
            %计算群延迟
            GD = zeros(N,L);
            for k = 1:N
                for m = 1:L
                    GD(k,m) = t(k)-1j*(dWx(k,m)./Wx(k,m));
                end
            end
            GD(abs(Wx)<gamma)=0;
            %重排
            for k = 1:N
                for m = 1:L
                    time = min(max(1 + round((real(GD(k,m))-t(1))/dt),1),N);
                    fre = min(max(1 + round((real(Ifreq(k,m))-f(1))/df),1),L);
                    TFx(time, fre) = TFx(time, fre) + Wx(k,m)*conj(Wx(k,m))*dt*df;
                end
            end
%%%%%%%%%%%%%计算MRM%%%%%%%%%%%%%%%%
        elseif (strcmp(tf_type, 'MRM'))
            %G(w)
            gt = @(t) s^(-1/2)*pi^(-1/4).*exp(-t.^2/s^2/2);
            Wx = zeros(N,L);
            for ptr = 1:N
                gh = gt(t-tao(ptr));
                gh = conj(gh);
                xcpsi = fft(gh .* x);
                Wx(ptr,:) = xcpsi(1:L);
            end
            %DG(w)
            gt = @(t) (-1j*t)*s^(-1/2)*pi^(-1/4).*exp(-t.^2/s^2/2);
            dWx = zeros(N,L);
            for ptr = 1:N
                gh = gt(t-tao(ptr));
                gh = conj(gh);
                xcpsi = fft(gh .* x);
                dWx(ptr,:) = xcpsi(1:L);
            end
            %wG(w)
            gt = @(t) (1j*t)/s/s*s^(-1/2)*pi^(-1/4).*exp(-t.^2/s^2/2);
            wWx = zeros(N,L);
            for ptr = 1:N
                gh = gt(t-tao(ptr));
                gh = conj(gh);
                xcpsi = fft(gh .* x);
                wWx(ptr,:) = xcpsi(1:L);
            end
            %计算瞬时频率
            Ifreq = zeros(N,L);
            for k = 1:N
                for m = 1:L
                    Ifreq(k,m) = 2*pi*f(m)+conj(wWx(k,m)./Wx(k,m));
                end
            end
            Ifreq = Ifreq/2/pi;
            Ifreq(abs(Wx)<gamma)=0;
            %计算群延迟
            GD = zeros(N,L);
            for k = 1:N
                for m = 1:L
                    GD(k,m) = t(k)-1j*(dWx(k,m)./Wx(k,m));
                end
            end
            GD(abs(Wx)<gamma)=0;
            %重排
            flag = zeros(N,L);
            for k = 1:N
                for m = 1:L
                    time = min(max(1 + round((real(GD(k,m))-t(1))/dt),1),N);
                    fre = min(max(1 + round((real(Ifreq(k,m))-f(1))/df),1),L);
                    if ((time == k || fre == m)&&flag(time,fre)==0)
                        if(time == k && fre == m)
                            TFx(time,fre) = Wx(k,m);
                            flag(time,fre) = 1;
                        else
                            TFx(time,fre) = Wx(k,m);
                        end
                    end
                end
            end
%%%%%%%%%%%%%计算SST1%%%%%%%%%%%%%%%%
        elseif (strcmp(tf_type, 'SST1'))
            %G(w)
            gt = @(t) s^(-1/2)*pi^(-1/4).*exp(-t.^2/s^2/2);
            Wx = zeros(N,L);
            for ptr = 1:N
                gh = gt(t-tao(ptr));
                gh = conj(gh);
                xcpsi = fft(gh .* x);
                Wx(ptr,:) = xcpsi(1:L);
            end
            %wG(w)
            gt = @(t) (1j*t)/s/s*s^(-1/2)*pi^(-1/4).*exp(-t.^2/s^2/2);
            wWx = zeros(N,L);
            for ptr = 1:N
                gh = gt(t-tao(ptr));
                gh = conj(gh);
                xcpsi = fft(gh .* x);
                wWx(ptr,:) = xcpsi(1:L);
            end
            %计算瞬时频率
            Ifreq = zeros(N,L);
            for k = 1:N
                for m = 1:L
                    Ifreq(k,m) = 2*pi*f(m)+conj(wWx(k,m)./Wx(k,m));
                end
            end
            Ifreq = Ifreq/2/pi;
            Ifreq(abs(Wx)<gamma)=0;
            %重排
            for k = 1:N
                for m = 1:L
                    fre = min(max(1 + round((real(Ifreq(k,m))-f(1))/df),1),L);
                    TFx(k, fre) = TFx(k, fre) + Wx(k,m)*df;
                end
            end
            warning('仅用于对比分析，请勿实际工程应用中使用！') 
%%%%%%%%%%%%%计算TSST1%%%%%%%%%%%%%%%%
        elseif (strcmp(tf_type, 'TSST1'))
            %G(w)
            gt = @(t) s^(-1/2)*pi^(-1/4).*exp(-t.^2/s^2/2);
            Wx = zeros(N,L);
            for ptr = 1:N
                gh = gt(t-tao(ptr));
                gh = conj(gh);
                xcpsi = fft(gh .* x);
                Wx(ptr,:) = xcpsi(1:L);
            end
            %DG(w)
            gt = @(t) (-1j*t)*s^(-1/2)*pi^(-1/4).*exp(-t.^2/s^2/2);
            dWx = zeros(N,L);
            for ptr = 1:N
                gh = gt(t-tao(ptr));
                gh = conj(gh);
                xcpsi = fft(gh .* x);
                dWx(ptr,:) = xcpsi(1:L);
            end
            %计算群延迟
            GD = zeros(N,L);
            for k = 1:N
                for m = 1:L
                    GD(k,m) = t(k)-1j*(dWx(k,m)./Wx(k,m));
                end
            end
            GD(abs(Wx)<gamma)=0;
            %重排
            for k = 1:N
                for m = 1:L
                    time = min(max(1 + round((real(GD(k,m))-t(1))/dt),1),N);
                    TFx(time, m) = TFx(time, m) + Wx(k,m)*dt;
                end
            end  
%%%%%%%%%%%%%计算TSST2%%%%%%%%%%%%%%%%
        elseif (strcmp(tf_type, 'TSST2'))
            %G(w)
            gt = @(t) s^(-1/2)*pi^(-1/4).*exp(-t.^2/s^2/2);
            Wx = zeros(N,L);
            for ptr = 1:N
                gh = gt(t-tao(ptr));
                gh = conj(gh);
                xcpsi = fft(gh .* x);
                Wx(ptr,:) = xcpsi(1:L);
            end
            %DG(w)
            gt = @(t) (-1j*t)*s^(-1/2)*pi^(-1/4).*exp(-t.^2/s^2/2);
            dWx = zeros(N,L);
            for ptr = 1:N
                gh = gt(t-tao(ptr));
                gh = conj(gh);
                xcpsi = fft(gh .* x);
                dWx(ptr,:) = xcpsi(1:L);
            end
            %wG(w)
            gt = @(t) (1j*t)/s/s*s^(-1/2)*pi^(-1/4).*exp(-t.^2/s^2/2);
            wWx = zeros(N,L);
            for ptr = 1:N
                gh = gt(t-tao(ptr));
                gh = conj(gh);
                xcpsi = fft(gh .* x);
                wWx(ptr,:) = xcpsi(1:L);
            end
            %DDG(w)
            gt = @(t) (-t.*t).*s^(-1/2).*pi^(-1/4).*exp(-t.^2/s^2/2);
            ddWx = zeros(N,L);
            for ptr = 1:N
                gh = gt(t-tao(ptr));
                gh = conj(gh);
                xcpsi = fft(gh .* x);
                ddWx(ptr,:) = xcpsi(1:L);
            end
            %wDG(w)
            gt = @(t) ((t.*t/s/s)-1).*s^(-1/2).*pi^(-1/4).*exp(-t.^2/s^2/2);
            wdWx = zeros(N,L);
            for ptr = 1:N
                gh = gt(t-tao(ptr));
                gh = conj(gh);
                xcpsi = fft(gh .* x);
                wdWx(ptr,:) = xcpsi(1:L);
            end
            %计算群延迟
            Denominator = wdWx.*Wx-dWx.*wWx;
            Numerator = ddWx.*wWx-dWx.*wdWx;
            Numerator2 = dWx.*dWx - ddWx.*Wx;
            p = Numerator./Denominator;
            q = Numerator2./Denominator;
            for ptr = 1:N
                p(ptr,:) = p(ptr,:) - 1i*t(ptr);
            end
            Rep = real(p);
            GD = -imag(p);
            GD(abs(Denominator)<gamma)=0;
            q(abs(Denominator)<gamma)=0;
            Chirprate = -imag(q);
            %重排
            for k = 1:N
                for m = 1:L
                    time = min(max(1 + round((real(GD(k,m))-t(1))/dt),1),N);
                    TFx(time, m) = TFx(time, m) + Wx(k,m)*dt;
                end
            end  
%%%%%%%%%%%%%计算TSET1%%%%%%%%%%%%%%%%            
        elseif (strcmp(tf_type, 'TSET1'))
            %G(w)
            gt = @(t) s^(-1/2)*pi^(-1/4).*exp(-t.^2/s^2/2);
            Wx = zeros(N,L);
            for ptr = 1:N
                gh = gt(t-tao(ptr));
                gh = conj(gh);
                xcpsi = fft(gh .* x);
                Wx(ptr,:) = xcpsi(1:L);
            end
            %DG(w)
            gt = @(t) (-1j*t)*s^(-1/2)*pi^(-1/4).*exp(-t.^2/s^2/2);
            dWx = zeros(N,L);
            for ptr = 1:N
                gh = gt(t-tao(ptr));
                gh = conj(gh);
                xcpsi = fft(gh .* x);
                dWx(ptr,:) = xcpsi(1:L);
            end
            %计算群延迟
            GD = zeros(N,L);
            for k = 1:N
                for m = 1:L
                    GD(k,m) = t(k)-1j*(dWx(k,m)./Wx(k,m));
                end
            end
            GD(abs(Wx)<gamma)=0;
            %提取
            for k = 1:N
                for m = 1:L
                    time = min(max(1 + round((real(GD(k,m))-t(1))/dt),1),N);
                    if (time == k)
                        TFx(time, m) = Wx(k,m);
                    end
                end
            end 
%%%%%%%%%%%%%计算TSET2%%%%%%%%%%%%%%%%            
        elseif (strcmp(tf_type, 'TSET2'))
            %G(w)
            gt = @(t) s^(-1/2)*pi^(-1/4).*exp(-t.^2/s^2/2);
            Wx = zeros(N,L);
            for ptr = 1:N
                gh = gt(t-tao(ptr));
                gh = conj(gh);
                xcpsi = fft(gh .* x);
                Wx(ptr,:) = xcpsi(1:L);
            end
            %DG(w)
            gt = @(t) (-1j*t)*s^(-1/2)*pi^(-1/4).*exp(-t.^2/s^2/2);
            dWx = zeros(N,L);
            for ptr = 1:N
                gh = gt(t-tao(ptr));
                gh = conj(gh);
                xcpsi = fft(gh .* x);
                dWx(ptr,:) = xcpsi(1:L);
            end
            %wG(w)
            gt = @(t) (1j*t)/s/s*s^(-1/2)*pi^(-1/4).*exp(-t.^2/s^2/2);
            wWx = zeros(N,L);
            for ptr = 1:N
                gh = gt(t-tao(ptr));
                gh = conj(gh);
                xcpsi = fft(gh .* x);
                wWx(ptr,:) = xcpsi(1:L);
            end
            %DDG(w)
            gt = @(t) (-t.*t).*s^(-1/2).*pi^(-1/4).*exp(-t.^2/s^2/2);
            ddWx = zeros(N,L);
            for ptr = 1:N
                gh = gt(t-tao(ptr));
                gh = conj(gh);
                xcpsi = fft(gh .* x);
                ddWx(ptr,:) = xcpsi(1:L);
            end
            %wDG(w)
            gt = @(t) ((t.*t/s/s)-1).*s^(-1/2).*pi^(-1/4).*exp(-t.^2/s^2/2);
            wdWx = zeros(N,L);
            for ptr = 1:N
                gh = gt(t-tao(ptr));
                gh = conj(gh);
                xcpsi = fft(gh .* x);
                wdWx(ptr,:) = xcpsi(1:L);
            end
            %计算群延迟
            Denominator = wdWx.*Wx-dWx.*wWx;
            Numerator = ddWx.*wWx-dWx.*wdWx;
            Numerator2 = dWx.*dWx - ddWx.*Wx;
            p = Numerator./Denominator;
            q = Numerator2./Denominator;
            for ptr = 1:N
                p(ptr,:) = p(ptr,:) - 1i*t(ptr);
            end
            Rep = real(p);
            GD = -imag(p);
            GD(abs(Denominator)<gamma)=0;
            q(abs(Denominator)<gamma)=0;
            Chirprate = -imag(q);
            %提取
            for k = 1:N
                for m = 1:L
                    time = min(max(1 + round((real(GD(k,m))-t(1))/dt),1),N);
                    if (time == k)
                        TFx(time, m) = Wx(k,m);
                    end
                end
            end  
%%%%%%%%%%%%%其它%%%%%%%%%%%%%%%%
        else
            error([stft_type,'不支持：',tf_type]);
        end        
%% Modified时频计算
    else
        %%%%%%%%%%%%%计算公共部分%%%%%%%%%%%%%%%%
        N = length(x);
        t = (0:N-1)/fs;
        fftx = fftshift(fft(x));
        delta_w = 2*pi/N;
        if mod(N,2)==0
            w = -pi+delta_w*(0:N-1);
            L = N/2+1;
        else
            w = -pi+delta_w/2+delta_w*(0:N-1);
            L = (N+1)/2;
        end   
        Omega = w*fs;%数字频率转模拟频率   
        f = (0:L-1)*delta_w*fs;%频移参数（rad）
        TFx = zeros(N,L);
        dt = 1/fs;
        df = (f(2)-f(1))/2/pi;
%%%%%%%%%%%%%计算STFT%%%%%%%%%%%%%%%%
        if (strcmp(tf_type, 'STFT'))
            gf = @(Omega) sqrt(2*s)*pi^(1/4).*exp(-(s*Omega).^2/2);
            Wx = zeros(N,L);
            for ptr = 1:L
                gh = gf(Omega-f(ptr));
                gh = conj(gh);
                xcpsi = ifft(ifftshift(gh .* fftx));
                Wx(:,ptr) = xcpsi;
            end
            TFx = Wx;
            f = f/2/pi;
%%%%%%%%%%%%%计算RM%%%%%%%%%%%%%%%%
        elseif (strcmp(tf_type, 'RM'))
            %G(w)
            gf = @(Omega) sqrt(2*s)*pi^(1/4).*exp(-(s*Omega).^2/2);
            Wx = zeros(N,L);
            for ptr = 1:L
                gh = gf(Omega-f(ptr));
                gh = conj(gh);
                xcpsi = ifft(ifftshift(gh .* fftx));
                Wx(:,ptr) = xcpsi;
            end
            %DG(w)
            gf = @(Omega) -Omega.*s*s*sqrt(2*s)*pi^(1/4).*exp(-(s*Omega).^2/2);
            dWx = zeros(N,L);
            for ptr = 1:L
                gh = gf(Omega-f(ptr));
                gh = conj(gh);
                xcpsi = ifft(ifftshift(gh .* fftx));
                dWx(:,ptr) = xcpsi;
            end
            %wG(w)
            gf = @(Omega) Omega.*sqrt(2*s)*pi^(1/4).*exp(-(s*Omega).^2/2);
            wWx = zeros(N,L);
            for ptr = 1:L
                gh = gf(Omega-f(ptr));
                gh = conj(gh);
                xcpsi = ifft(ifftshift(gh .* fftx));
                wWx(:,ptr) = xcpsi;
            end
            %计算瞬时频率
            Ifreq = zeros(N,L);
            for k = 1:N
                for m = 1:L
                    Ifreq(k,m) = f(m)+conj(wWx(k,m)./Wx(k,m));
                end
            end
            Ifreq = Ifreq/2/pi;
            Ifreq(abs(Wx)<gamma)=0;
            %计算群延迟
            GD = zeros(N,L);
            for k = 1:N
                for m = 1:L
                    GD(k,m) = t(k)-1j*(dWx(k,m)./Wx(k,m));
                end
            end
            GD(abs(Wx)<gamma)=0;
            %重排
            f = f/2/pi;
            for k = 1:N
                for m = 1:L
                    time = min(max(1 + round((real(GD(k,m))-t(1))/dt),1),N);
                    fre = min(max(1 + round((real(Ifreq(k,m))-f(1))/df),1),L);
                    TFx(time, fre) = TFx(time, fre) + Wx(k,m)*conj(Wx(k,m))*dt*df;
                end
            end
%%%%%%%%%%%%%计算MRM%%%%%%%%%%%%%%%%
        elseif (strcmp(tf_type, 'MRM'))
            %G(w)
            gf = @(Omega) sqrt(2*s)*pi^(1/4).*exp(-(s*Omega).^2/2);
            Wx = zeros(N,L);
            for ptr = 1:L
                gh = gf(Omega-f(ptr));
                gh = conj(gh);
                xcpsi = ifft(ifftshift(gh .* fftx));
                Wx(:,ptr) = xcpsi;
            end
            %DG(w)
            gf = @(Omega) -Omega.*s*s*sqrt(2*s)*pi^(1/4).*exp(-(s*Omega).^2/2);
            dWx = zeros(N,L);
            for ptr = 1:L
                gh = gf(Omega-f(ptr));
                gh = conj(gh);
                xcpsi = ifft(ifftshift(gh .* fftx));
                dWx(:,ptr) = xcpsi;
            end
            %wG(w)
            gf = @(Omega) Omega.*sqrt(2*s)*pi^(1/4).*exp(-(s*Omega).^2/2);
            wWx = zeros(N,L);
            for ptr = 1:L
                gh = gf(Omega-f(ptr));
                gh = conj(gh);
                xcpsi = ifft(ifftshift(gh .* fftx));
                wWx(:,ptr) = xcpsi;
            end
            %计算瞬时频率
            Ifreq = zeros(N,L);
            for k = 1:N
                for m = 1:L
                    Ifreq(k,m) = f(m)+conj(wWx(k,m)./Wx(k,m));
                end
            end
            Ifreq = Ifreq/2/pi;
            Ifreq(abs(Wx)<gamma)=0;
            %计算群延迟
            GD = zeros(N,L);
            for k = 1:N
                for m = 1:L
                    GD(k,m) = t(k)-1j*(dWx(k,m)./Wx(k,m));
                end
            end
            GD(abs(Wx)<gamma)=0;
            %重排
            f = f/2/pi;
            flag = zeros(N,L);
            for k = 1:N
                for m = 1:L
                    time = min(max(1 + round((real(GD(k,m))-t(1))/dt),1),N);
                    fre = min(max(1 + round((real(Ifreq(k,m))-f(1))/df),1),L);
                    if ((time == k || fre == m)&&flag(time,fre)==0)
                        if(time == k && fre == m)
                            TFx(time,fre) = Wx(k,m);
                            flag(time,fre) = 1;
                        else
                            TFx(time,fre) = Wx(k,m);
                        end
                    end
                end
            end
%%%%%%%%%%%%%计算SST1%%%%%%%%%%%%%%%%
        elseif (strcmp(tf_type, 'SST1'))
            %G(w)
            gf = @(Omega) sqrt(2*s)*pi^(1/4).*exp(-(s*Omega).^2/2);
            Wx = zeros(N,L);
            for ptr = 1:L
                gh = gf(Omega-f(ptr));
                gh = conj(gh);
                xcpsi = ifft(ifftshift(gh .* fftx));
                Wx(:,ptr) = xcpsi;
            end
            %wG(w)
            gf = @(Omega) Omega.*sqrt(2*s)*pi^(1/4).*exp(-(s*Omega).^2/2);
            wWx = zeros(N,L);
            for ptr = 1:L
                gh = gf(Omega-f(ptr));
                gh = conj(gh);
                xcpsi = ifft(ifftshift(gh .* fftx));
                wWx(:,ptr) = xcpsi;
            end
            %计算瞬时频率
            Ifreq = zeros(N,L);
            for k = 1:N
                for m = 1:L
                    Ifreq(k,m) = f(m)+conj(wWx(k,m)./Wx(k,m));
                end
            end
            Ifreq = Ifreq/2/pi;
            Ifreq(abs(Wx)<gamma)=0;
            %重排
            f = f/2/pi;
            for k = 1:N
                for m = 1:L
                    fre = min(max(1 + round((real(Ifreq(k,m))-f(1))/df),1),L);
                    TFx(k, fre) = TFx(k, fre) + Wx(k,m)*df;
                end
            end
%%%%%%%%%%%%%计算TSST1%%%%%%%%%%%%%%%%
        elseif (strcmp(tf_type, 'TSST1'))
            %G(w)
            gf = @(Omega) sqrt(2*s)*pi^(1/4).*exp(-(s*Omega).^2/2);
            Wx = zeros(N,L);
            for ptr = 1:L
                gh = gf(Omega-f(ptr));
                gh = conj(gh);
                xcpsi = ifft(ifftshift(gh .* fftx));
                Wx(:,ptr) = xcpsi;
            end
            %DG(w)
            gf = @(Omega) -Omega.*s*s*sqrt(2*s)*pi^(1/4).*exp(-(s*Omega).^2/2);
            dWx = zeros(N,L);
            for ptr = 1:L
                gh = gf(Omega-f(ptr));
                gh = conj(gh);
                xcpsi = ifft(ifftshift(gh .* fftx));
                dWx(:,ptr) = xcpsi;
            end
            %计算群延迟
            GD = zeros(N,L);
            for k = 1:N
                for m = 1:L
                    GD(k,m) = t(k)-1j*(dWx(k,m)./Wx(k,m));
                end
            end
            GD(abs(Wx)<gamma)=0;
            %重排
            f = f/2/pi;
            for k = 1:N
                for m = 1:L
                    time = min(max(1 + round((real(GD(k,m))-t(1))/dt),1),N);
                    TFx(time, m) = TFx(time, m) + Wx(k,m)*dt;
                end
            end
            warning('仅用于对比分析，请勿实际工程应用中使用！') 
%%%%%%%%%%%%%计算SST2%%%%%%%%%%%%%%%%
        elseif (strcmp(tf_type, 'SST2'))
            %g(t)
            gf = @(Omega) sqrt(2*s)*pi^(1/4).*exp(-(s*Omega).^2/2);
            Wx = zeros(N,L);
            for ptr = 1:L
                gh = gf(Omega-f(ptr));
                gh = conj(gh);
                xcpsi = ifft(ifftshift(gh .* fftx));
                Wx(:,ptr) = xcpsi;
            end
            %dg(t)
            gf = @(Omega) (1j*Omega).*sqrt(2*s)*pi^(1/4).*exp(-(s*Omega).^2/2);
            dWx = zeros(N,L);
            for ptr = 1:L
                gh = gf(Omega-f(ptr));
                gh = conj(gh);
                xcpsi = ifft(ifftshift(gh .* fftx));
                dWx(:,ptr) = xcpsi;
            end
            %ddg(t)
            gf = @(Omega) -(Omega.*Omega).*sqrt(2*s)*pi^(1/4).*exp(-(s*Omega).^2/2);
            ddWx = zeros(N,L);
            for ptr = 1:L
                gh = gf(Omega-f(ptr));
                gh = conj(gh);
                xcpsi = ifft(ifftshift(gh .* fftx));
                ddWx(:,ptr) = xcpsi;
            end
            %tg(t)
            gf = @(Omega) -1j*Omega.*s*s*sqrt(2*s)*pi^(1/4).*exp(-(s*Omega).^2/2);
            tWx = zeros(N,L);
            for ptr = 1:L
                gh = gf(Omega-f(ptr));
                gh = conj(gh);
                xcpsi = ifft(ifftshift(gh .* fftx));
                tWx(:,ptr) = xcpsi;
            end
            %tdg(t)
            gf = @(Omega) (s*s*Omega.*Omega-1).*sqrt(2*s)*pi^(1/4).*exp(-(s*Omega).^2/2);
            tdWx = zeros(N,L);
            for ptr = 1:L
                gh = gf(Omega-f(ptr));
                gh = conj(gh);
                xcpsi = ifft(ifftshift(gh .* fftx));
                tdWx(:,ptr) = xcpsi;
            end
            %计算瞬时频率
            Denominator = tdWx.*Wx-tWx.*dWx;
            Numerator = ddWx.*tWx-dWx.*tdWx;
            Numerator2 = dWx.*dWx-ddWx.*Wx;
            p = Numerator./Denominator;
            q = Numerator2./Denominator;
            for ptr = 1:L
                p(:,ptr) = p(:,ptr) + 1i*f(ptr);
            end
            Ifreq = imag(p)/2/pi;
            Ifreq(abs(Denominator)<gamma)=0;
            q(abs(Denominator)<gamma)=0;
            Chirprate = imag(q);
            %重排
            f = f/2/pi;
            for k = 1:N
                for m = 1:L
                    fre = min(max(1 + round((real(Ifreq(k,m))-f(1))/df),1),L);
                    TFx(k, fre) = TFx(k, fre) + Wx(k,m)*df;
                end
            end
%%%%%%%%%%%%%计算SET1%%%%%%%%%%%%%%%%            
        elseif (strcmp(tf_type, 'SET1'))
            %G(w)
            gf = @(Omega) sqrt(2*s)*pi^(1/4).*exp(-(s*Omega).^2/2);
            Wx = zeros(N,L);
            for ptr = 1:L
                gh = gf(Omega-f(ptr));
                gh = conj(gh);
                xcpsi = ifft(ifftshift(gh .* fftx));
                Wx(:,ptr) = xcpsi;
            end
            %wG(w)
            gf = @(Omega) Omega.*sqrt(2*s)*pi^(1/4).*exp(-(s*Omega).^2/2);
            wWx = zeros(N,L);
            for ptr = 1:L
                gh = gf(Omega-f(ptr));
                gh = conj(gh);
                xcpsi = ifft(ifftshift(gh .* fftx));
                wWx(:,ptr) = xcpsi;
            end
            %计算瞬时频率
            Ifreq = zeros(N,L);
            for k = 1:N
                for m = 1:L
                    Ifreq(k,m) = f(m)+conj(wWx(k,m)./Wx(k,m));
                end
            end
            Ifreq = Ifreq/2/pi;
            Ifreq(abs(Wx)<gamma)=0;
            %重排
            f = f/2/pi;
            for k = 1:N
                for m = 1:L
                    fre = min(max(1 + round((real(Ifreq(k,m))-f(1))/df),1),L);
                    if (fre == m)
                        TFx(k, fre) = Wx(k,m);
                    end
                end
            end
%%%%%%%%%%%%%计算SET2%%%%%%%%%%%%%%%%            
        elseif (strcmp(tf_type, 'SET2'))
            %g(t)
            gf = @(Omega) sqrt(2*s)*pi^(1/4).*exp(-(s*Omega).^2/2);
            Wx = zeros(N,L);
            for ptr = 1:L
                gh = gf(Omega-f(ptr));
                gh = conj(gh);
                xcpsi = ifft(ifftshift(gh .* fftx));
                Wx(:,ptr) = xcpsi;
            end
            %dg(t)
            gf = @(Omega) (1j*Omega).*sqrt(2*s)*pi^(1/4).*exp(-(s*Omega).^2/2);
            dWx = zeros(N,L);
            for ptr = 1:L
                gh = gf(Omega-f(ptr));
                gh = conj(gh);
                xcpsi = ifft(ifftshift(gh .* fftx));
                dWx(:,ptr) = xcpsi;
            end
            %ddg(t)
            gf = @(Omega) -(Omega.*Omega).*sqrt(2*s)*pi^(1/4).*exp(-(s*Omega).^2/2);
            ddWx = zeros(N,L);
            for ptr = 1:L
                gh = gf(Omega-f(ptr));
                gh = conj(gh);
                xcpsi = ifft(ifftshift(gh .* fftx));
                ddWx(:,ptr) = xcpsi;
            end
            %tg(t)
            gf = @(Omega) -1j*Omega.*s*s*sqrt(2*s)*pi^(1/4).*exp(-(s*Omega).^2/2);
            tWx = zeros(N,L);
            for ptr = 1:L
                gh = gf(Omega-f(ptr));
                gh = conj(gh);
                xcpsi = ifft(ifftshift(gh .* fftx));
                tWx(:,ptr) = xcpsi;
            end
            %tdg(t)
            gf = @(Omega) (s*s*Omega.*Omega-1).*sqrt(2*s)*pi^(1/4).*exp(-(s*Omega).^2/2);
            tdWx = zeros(N,L);
            for ptr = 1:L
                gh = gf(Omega-f(ptr));
                gh = conj(gh);
                xcpsi = ifft(ifftshift(gh .* fftx));
                tdWx(:,ptr) = xcpsi;
            end
            %计算瞬时频率
            Denominator = tdWx.*Wx-tWx.*dWx;
            Numerator = ddWx.*tWx-dWx.*tdWx;
            Numerator2 = dWx.*dWx-ddWx.*Wx;
            p = Numerator./Denominator;
            q = Numerator2./Denominator;
            for ptr = 1:L
                p(:,ptr) = p(:,ptr) + 1i*f(ptr);
            end
            Rep = real(p);
            Ifreq = imag(p)/2/pi;
            Ifreq(abs(Denominator)<gamma)=0;
            q(abs(Denominator)<gamma)=0;
            Chirprate = imag(q);
            %重排
            f = f/2/pi;
            for k = 1:N
                for m = 1:L
                    fre = min(max(1 + round((real(Ifreq(k,m))-f(1))/df),1),L);
                    if (fre == m)
                        TFx(k, fre) = Wx(k,m);
                    end
                end
            end        
%%%%%%%%%%%%%其它%%%%%%%%%%%%%%%%
        else
            error([stft_type,'不支持：',tf_type]);
        end    
    end
end