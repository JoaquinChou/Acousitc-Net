function [Wx,TFx,Rep_t,Rep_m,q_t,q_m,t,f] = DET(x,fs,s,gamma)
%% 参数说明
%函数名：双重提取变换；英文名：Dual-Extraction Transform（DET）
%%%%%%%%%%%%%%输入参数%%%%%%%%%%%%%%
%x:信号向量
%fs:采样率
%s:窗长度
%gamma:阈值
%%%%%%%%%%%%%%输出参数%%%%%%%%%%%%%%
%Wx：短时傅里叶的TFR
%TFx：改进重排变换的TFR
%Rep_t、q_t：时间重构因子
%Rep_m、q_m：频率重构因子
%t:时间序列
%f:频率序列
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 预处理
%判断是否为列向量，是则转置
    [xrow,~] = size(x);
    if (xrow~=1)
        x = x';
    end
%计算公共部分
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
%% 计算各个短时傅里叶
    %G(w)
    gt = @(t) s^(-1/2)*pi^(-1/4).*exp(-t.^2/s^2/2);
    G = zeros(N,L);
    for ptr = 1:N
        gh = gt(t-tao(ptr));
        gh = conj(gh);
        xcpsi = fft(gh .* x);
        G(ptr,:) = xcpsi(1:L);
    end
    %DG(w)
    gt = @(t) (-1j*t)*s^(-1/2)*pi^(-1/4).*exp(-t.^2/s^2/2);
    DG = zeros(N,L);
    for ptr = 1:N
        gh = gt(t-tao(ptr));
        gh = conj(gh);
        xcpsi = fft(gh .* x);
        DG(ptr,:) = xcpsi(1:L);
    end
    %wG(w)
    wG = DG*(-1/s^2);
    %DDG(w)
    gt = @(t) (-t.*t).*s^(-1/2).*pi^(-1/4).*exp(-t.^2/s^2/2);
    DDG = zeros(N,L);
    for ptr = 1:N
        gh = gt(t-tao(ptr));
        gh = conj(gh);
        xcpsi = fft(gh .* x);
        DDG(ptr,:) = xcpsi(1:L);
    end
    %wDG(w)
    wDG = DDG*(-1/s^2)- G;
    %g(t)
    g = G;
    %dg(t)
    dg = 1j*wG;
    %ddg(t)
    ddg = wDG/s^2;
    %tg(t)
    tg = 1j*DG;
    %tdg(t)
    tdg = -(G+wDG);
%% 计算算子  
    %计算瞬时频率
    Ifreq = zeros(N,L);
    for k = 1:N
        for m = 1:L
            Ifreq(k,m) = 2*pi*f(m)+conj(wG(k,m)./G(k,m));
        end
    end
    Ifreq = Ifreq/2/pi;
    Ifreq(abs(G)<gamma)=0;
    %计算群延迟
    GD = zeros(N,L);
    for k = 1:N
        for m = 1:L
            GD(k,m) = t(k)-1j*(DG(k,m)./G(k,m));
        end
    end
    GD(abs(G)<gamma)=0;    
%% 计算重构算子
%时间方向
    Denominator = wDG.*G-DG.*wG;
    Numerator = DDG.*wG-DG.*wDG;
    Numerator2 = DG.*DG - DDG.*G;
    p = Numerator./Denominator;
    q_t = Numerator2./Denominator;
    for ptr = 1:N
        p(ptr,:) = p(ptr,:) - 1i*t(ptr);
    end
    Rep_t = real(p);
    q_t(abs(Denominator)<gamma)=0; 
    Rep_t(abs(Denominator)<gamma)=0; 
%频率方向
    Denominator = tdg.*g-tg.*dg;
    Numerator = ddg.*tg-dg.*tdg;
    Numerator2 = dg.*dg-ddg.*g;
    p = Numerator./Denominator;
    q_m = Numerator2./Denominator;
    for ptr = 1:L
        p(:,ptr) = p(:,ptr) + 1i*f(ptr);
    end
    Rep_m = real(p);   
    q_m(abs(Denominator)<gamma)=0; 
    Rep_m(abs(Denominator)<gamma)=0; 
%% 重排
    flag = zeros(N,L);
    for k = 1:N
        for m = 1:L
            time = min(max(1 + round((real(GD(k,m))-t(1))/dt),1),N);
            fre = min(max(1 + round((real(Ifreq(k,m))-f(1))/df),1),L);
            if ((time == k || fre == m)&&flag(time,fre)==0)
            if ((time == k || fre == m))
                if(time == k && fre == m)
                    TFx(time,fre) = G(k,m);
                    flag(time,fre) = 1;
                else
                    TFx(time,fre) = G(k,m);
                end
            end
        end
    end
    Wx = G;
end