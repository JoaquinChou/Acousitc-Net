function [Wx,TFx,Ifreq,GD,Rep,Chirprate,q,t,f] = TFM(x,fs,s,stft_type,tf_type,gamma)
%% ����˵��
%��������ʱƵ����������Ӣ������Time-Frequency Method��TFM��
%%%%%%%%%%%%%%�������%%%%%%%%%%%%%%
%x:�ź�����
%fs:������
%s:������
%stft_type:���ö�ʱ����Ҷ�任������
%tf_type:ʱƵ�任����
%gamma:��ֵ
%%%%%%%%%%%%%%�������%%%%%%%%%%%%%%
%Wx����ʱ����Ҷ��TFR
%TFx���Ľ����ű任��TFR
%Ifreq��GD��˲ʱƵ�������Լ�Ⱥ�ӳ�����
%Chirprate��Chirprate����
%Rep��q���ع�����
%t:ʱ������
%f:Ƶ������
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%��֧�ֵ�ʱƵ������
%ʱ�䷽��
%STFT :��ʱ����Ҷ�任
%RM   :���ű任
%TSST1:һ��ʱ��ͬ��ѹ���任��
%TSST2:����ʱ��ͬ��ѹ���任��
%TSET1:һ��ʱ��ͬ����ȡ�任��
%TSST2:����ʱ��ͬ����ȡ�任��
%MRM  :�Ľ����ű任��
%SST1 :һ��Ƶ��ͬ��ѹ���任(�����ڶԱ�)��
%Ƶ�ʷ���
%STFT :��ʱ����Ҷ�任
%RM   :���ű任
%SST1:һ��Ƶ��ͬ��ѹ���任��
%SST2:����Ƶ��ͬ��ѹ���任��
%SET1:һ��Ƶ��ͬ����ȡ�任��
%SST2:����Ƶ��ͬ����ȡ�任��
%MRM   :�Ľ����ű任��
%TSST1 :һ��ʱ��ͬ��ѹ���任(�����ڶԱ�)��
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ������Ŀ���
    if (nargin > 6)
        error('����������࣡');
    elseif(nargin == 5)
        gamma = 0;  
    elseif(nargin == 4)
        gamma = 0; tf_type = 'STFT';
    elseif(nargin == 3)
        gamma = 0; tf_type = 'STFT';stft_type = 'T';
    elseif(nargin == 2 || nargin == 1 || nargin == 0 )
        error('ȱ�����������');
    end
%stft_type���
    if ((~strcmp(stft_type, 'Traditional')) && (~strcmp(stft_type, 'T')) && ...
        (~strcmp(stft_type, 'Modified')) && (~strcmp(stft_type, 'M')))
        error(['stft_type��֧�֣�',stft_type,'���ͣ�',...
            '��������֧�֣�Traditional��T��Modified��M']);
    end
%tf_type���
    if ((~strcmp(tf_type, 'STFT')) && (~strcmp(tf_type, 'RM')) && ...
        (~strcmp(tf_type, 'SST1')) && (~strcmp(tf_type, 'SST2')) && ...
        (~strcmp(tf_type, 'SET1')) && (~strcmp(tf_type, 'SET2')) && ...
        (~strcmp(tf_type, 'TSST1')) && (~strcmp(tf_type, 'TSST2')) && ...
        (~strcmp(tf_type, 'TSET1')) && (~strcmp(tf_type, 'TSET2')) && (~strcmp(tf_type, 'MRM')))
        error(['tf_type��֧�֣�',tf_type,'���ͣ�',...
            '��������֧�֣�STFT��RM��SST1��SST2��SET1��SET2��TSST1��TSST2��TSET1��TSET2��MRM']);
    end
%�����ֵ
    Wx=0;TFx=0;Ifreq=0;GD=0;q=0;Rep=0;Chirprate=0;t=0;f=0;
%% Ԥ����
%�ж��Ƿ�Ϊ������������ת��
    [xrow,~] = size(x);
    if (xrow~=1)
        x = x';
    end
%% TraditionalʱƵ����
    if (strcmp(stft_type, 'Traditional') || strcmp(stft_type, 'T'))
%%%%%%%%%%%%%���㹫������%%%%%%%%%%%%%%%%
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
%%%%%%%%%%%%%����STFT%%%%%%%%%%%%%%%%
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
%%%%%%%%%%%%%����RM%%%%%%%%%%%%%%%%
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
            %����˲ʱƵ��
            Ifreq = zeros(N,L);
            for k = 1:N
                for m = 1:L
                    Ifreq(k,m) = 2*pi*f(m)+conj(wWx(k,m)./Wx(k,m));
                end
            end
            Ifreq = Ifreq/2/pi;
            Ifreq(abs(Wx)<gamma)=0;
            %����Ⱥ�ӳ�
            GD = zeros(N,L);
            for k = 1:N
                for m = 1:L
                    GD(k,m) = t(k)-1j*(dWx(k,m)./Wx(k,m));
                end
            end
            GD(abs(Wx)<gamma)=0;
            %����
            for k = 1:N
                for m = 1:L
                    time = min(max(1 + round((real(GD(k,m))-t(1))/dt),1),N);
                    fre = min(max(1 + round((real(Ifreq(k,m))-f(1))/df),1),L);
                    TFx(time, fre) = TFx(time, fre) + Wx(k,m)*conj(Wx(k,m))*dt*df;
                end
            end
%%%%%%%%%%%%%����MRM%%%%%%%%%%%%%%%%
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
            %����˲ʱƵ��
            Ifreq = zeros(N,L);
            for k = 1:N
                for m = 1:L
                    Ifreq(k,m) = 2*pi*f(m)+conj(wWx(k,m)./Wx(k,m));
                end
            end
            Ifreq = Ifreq/2/pi;
            Ifreq(abs(Wx)<gamma)=0;
            %����Ⱥ�ӳ�
            GD = zeros(N,L);
            for k = 1:N
                for m = 1:L
                    GD(k,m) = t(k)-1j*(dWx(k,m)./Wx(k,m));
                end
            end
            GD(abs(Wx)<gamma)=0;
            %����
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
%%%%%%%%%%%%%����SST1%%%%%%%%%%%%%%%%
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
            %����˲ʱƵ��
            Ifreq = zeros(N,L);
            for k = 1:N
                for m = 1:L
                    Ifreq(k,m) = 2*pi*f(m)+conj(wWx(k,m)./Wx(k,m));
                end
            end
            Ifreq = Ifreq/2/pi;
            Ifreq(abs(Wx)<gamma)=0;
            %����
            for k = 1:N
                for m = 1:L
                    fre = min(max(1 + round((real(Ifreq(k,m))-f(1))/df),1),L);
                    TFx(k, fre) = TFx(k, fre) + Wx(k,m)*df;
                end
            end
            warning('�����ڶԱȷ���������ʵ�ʹ���Ӧ����ʹ�ã�') 
%%%%%%%%%%%%%����TSST1%%%%%%%%%%%%%%%%
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
            %����Ⱥ�ӳ�
            GD = zeros(N,L);
            for k = 1:N
                for m = 1:L
                    GD(k,m) = t(k)-1j*(dWx(k,m)./Wx(k,m));
                end
            end
            GD(abs(Wx)<gamma)=0;
            %����
            for k = 1:N
                for m = 1:L
                    time = min(max(1 + round((real(GD(k,m))-t(1))/dt),1),N);
                    TFx(time, m) = TFx(time, m) + Wx(k,m)*dt;
                end
            end  
%%%%%%%%%%%%%����TSST2%%%%%%%%%%%%%%%%
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
            %����Ⱥ�ӳ�
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
            %����
            for k = 1:N
                for m = 1:L
                    time = min(max(1 + round((real(GD(k,m))-t(1))/dt),1),N);
                    TFx(time, m) = TFx(time, m) + Wx(k,m)*dt;
                end
            end  
%%%%%%%%%%%%%����TSET1%%%%%%%%%%%%%%%%            
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
            %����Ⱥ�ӳ�
            GD = zeros(N,L);
            for k = 1:N
                for m = 1:L
                    GD(k,m) = t(k)-1j*(dWx(k,m)./Wx(k,m));
                end
            end
            GD(abs(Wx)<gamma)=0;
            %��ȡ
            for k = 1:N
                for m = 1:L
                    time = min(max(1 + round((real(GD(k,m))-t(1))/dt),1),N);
                    if (time == k)
                        TFx(time, m) = Wx(k,m);
                    end
                end
            end 
%%%%%%%%%%%%%����TSET2%%%%%%%%%%%%%%%%            
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
            %����Ⱥ�ӳ�
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
            %��ȡ
            for k = 1:N
                for m = 1:L
                    time = min(max(1 + round((real(GD(k,m))-t(1))/dt),1),N);
                    if (time == k)
                        TFx(time, m) = Wx(k,m);
                    end
                end
            end  
%%%%%%%%%%%%%����%%%%%%%%%%%%%%%%
        else
            error([stft_type,'��֧�֣�',tf_type]);
        end        
%% ModifiedʱƵ����
    else
        %%%%%%%%%%%%%���㹫������%%%%%%%%%%%%%%%%
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
        Omega = w*fs;%����Ƶ��תģ��Ƶ��   
        f = (0:L-1)*delta_w*fs;%Ƶ�Ʋ�����rad��
        TFx = zeros(N,L);
        dt = 1/fs;
        df = (f(2)-f(1))/2/pi;
%%%%%%%%%%%%%����STFT%%%%%%%%%%%%%%%%
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
%%%%%%%%%%%%%����RM%%%%%%%%%%%%%%%%
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
            %����˲ʱƵ��
            Ifreq = zeros(N,L);
            for k = 1:N
                for m = 1:L
                    Ifreq(k,m) = f(m)+conj(wWx(k,m)./Wx(k,m));
                end
            end
            Ifreq = Ifreq/2/pi;
            Ifreq(abs(Wx)<gamma)=0;
            %����Ⱥ�ӳ�
            GD = zeros(N,L);
            for k = 1:N
                for m = 1:L
                    GD(k,m) = t(k)-1j*(dWx(k,m)./Wx(k,m));
                end
            end
            GD(abs(Wx)<gamma)=0;
            %����
            f = f/2/pi;
            for k = 1:N
                for m = 1:L
                    time = min(max(1 + round((real(GD(k,m))-t(1))/dt),1),N);
                    fre = min(max(1 + round((real(Ifreq(k,m))-f(1))/df),1),L);
                    TFx(time, fre) = TFx(time, fre) + Wx(k,m)*conj(Wx(k,m))*dt*df;
                end
            end
%%%%%%%%%%%%%����MRM%%%%%%%%%%%%%%%%
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
            %����˲ʱƵ��
            Ifreq = zeros(N,L);
            for k = 1:N
                for m = 1:L
                    Ifreq(k,m) = f(m)+conj(wWx(k,m)./Wx(k,m));
                end
            end
            Ifreq = Ifreq/2/pi;
            Ifreq(abs(Wx)<gamma)=0;
            %����Ⱥ�ӳ�
            GD = zeros(N,L);
            for k = 1:N
                for m = 1:L
                    GD(k,m) = t(k)-1j*(dWx(k,m)./Wx(k,m));
                end
            end
            GD(abs(Wx)<gamma)=0;
            %����
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
%%%%%%%%%%%%%����SST1%%%%%%%%%%%%%%%%
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
            %����˲ʱƵ��
            Ifreq = zeros(N,L);
            for k = 1:N
                for m = 1:L
                    Ifreq(k,m) = f(m)+conj(wWx(k,m)./Wx(k,m));
                end
            end
            Ifreq = Ifreq/2/pi;
            Ifreq(abs(Wx)<gamma)=0;
            %����
            f = f/2/pi;
            for k = 1:N
                for m = 1:L
                    fre = min(max(1 + round((real(Ifreq(k,m))-f(1))/df),1),L);
                    TFx(k, fre) = TFx(k, fre) + Wx(k,m)*df;
                end
            end
%%%%%%%%%%%%%����TSST1%%%%%%%%%%%%%%%%
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
            %����Ⱥ�ӳ�
            GD = zeros(N,L);
            for k = 1:N
                for m = 1:L
                    GD(k,m) = t(k)-1j*(dWx(k,m)./Wx(k,m));
                end
            end
            GD(abs(Wx)<gamma)=0;
            %����
            f = f/2/pi;
            for k = 1:N
                for m = 1:L
                    time = min(max(1 + round((real(GD(k,m))-t(1))/dt),1),N);
                    TFx(time, m) = TFx(time, m) + Wx(k,m)*dt;
                end
            end
            warning('�����ڶԱȷ���������ʵ�ʹ���Ӧ����ʹ�ã�') 
%%%%%%%%%%%%%����SST2%%%%%%%%%%%%%%%%
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
            %����˲ʱƵ��
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
            %����
            f = f/2/pi;
            for k = 1:N
                for m = 1:L
                    fre = min(max(1 + round((real(Ifreq(k,m))-f(1))/df),1),L);
                    TFx(k, fre) = TFx(k, fre) + Wx(k,m)*df;
                end
            end
%%%%%%%%%%%%%����SET1%%%%%%%%%%%%%%%%            
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
            %����˲ʱƵ��
            Ifreq = zeros(N,L);
            for k = 1:N
                for m = 1:L
                    Ifreq(k,m) = f(m)+conj(wWx(k,m)./Wx(k,m));
                end
            end
            Ifreq = Ifreq/2/pi;
            Ifreq(abs(Wx)<gamma)=0;
            %����
            f = f/2/pi;
            for k = 1:N
                for m = 1:L
                    fre = min(max(1 + round((real(Ifreq(k,m))-f(1))/df),1),L);
                    if (fre == m)
                        TFx(k, fre) = Wx(k,m);
                    end
                end
            end
%%%%%%%%%%%%%����SET2%%%%%%%%%%%%%%%%            
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
            %����˲ʱƵ��
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
            %����
            f = f/2/pi;
            for k = 1:N
                for m = 1:L
                    fre = min(max(1 + round((real(Ifreq(k,m))-f(1))/df),1),L);
                    if (fre == m)
                        TFx(k, fre) = Wx(k,m);
                    end
                end
            end        
%%%%%%%%%%%%%����%%%%%%%%%%%%%%%%
        else
            error([stft_type,'��֧�֣�',tf_type]);
        end    
    end
end