function [Wx2] = ConvertSTFT(Wx,t,f,Convert2type,gamma)
%% 参数说明
%函数名：反时频分析方法；英文名：Inverse Time-Frequency Method（ITFM）
%%%%%%%%%%%%%%输入参数%%%%%%%%%%%%%%
%Wx:时频表示
%t:时间序列
%f:频率序列
%Convert2type:转换到的短时傅里叶类型
%gamma:转换阈值
%%%%%%%%%%%%%%输出参数%%%%%%%%%%%%%%
%Wx2：转换后的时频表示
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 计算
    [N,L] = size(Wx);
    fs = 1/(t(2)-t(1));
    Wx2 = zeros(N,L);
    if strcmp(Convert2type, 'MSTFT') 
        for i = 1:N
            for j = 1:L
                if (abs(Wx(i,j))>gamma)
                    Wx2(i,j) = Wx(i,j)*exp(1i*2*pi*f(j)*t(i))/fs;
                end
            end
        end
    end
    if strcmp(Convert2type, 'TSTFT') 
        for i = 1:N
            for j = 1:L
                if (abs(Wx(i,j))>gamma)
                    Wx2(i,j) = Wx(i,j)*exp(-1i*2*pi*f(j)*t(i))*fs;
                end
            end
        end
    end 
end