function [Wx2] = ConvertSTFT(Wx,t,f,Convert2type,gamma)
%% ����˵��
%����������ʱƵ����������Ӣ������Inverse Time-Frequency Method��ITFM��
%%%%%%%%%%%%%%�������%%%%%%%%%%%%%%
%Wx:ʱƵ��ʾ
%t:ʱ������
%f:Ƶ������
%Convert2type:ת�����Ķ�ʱ����Ҷ����
%gamma:ת����ֵ
%%%%%%%%%%%%%%�������%%%%%%%%%%%%%%
%Wx2��ת�����ʱƵ��ʾ
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ����
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