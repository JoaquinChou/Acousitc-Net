function [Reconstruction,t] = ITFM(TFxi,fs,s,direction,tf_type)
%% ����˵��
%����������ʱƵ����������Ӣ������Inverse Time-Frequency Method��ITFM��
%%%%%%%%%%%%%%�������%%%%%%%%%%%%%%
%TFxi:��ʱƵ��ʾ
%fs:������
%s:������
%direction:������ȡ����
%tf_type:ʱƵ�任����
%%%%%%%%%%%%%%�������%%%%%%%%%%%%%%
%Reconstruction���ع��ź�
%t��ʱ������
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Ԥ����
    [N,~] = size(TFxi);
    t = (0:N-1)/fs;
    dt = 1/fs;
    df = fs/N;
%% ʱ�䷽��ѹ���ı任�ع�
    if (strcmp(direction, 'Time') || strcmp(direction, 'T'))
%%%%%%%%%%%%%���㹫������%%%%%%%%%%%%%%%%
        TimeSum = sum(TFxi,1);        
        mySpectrum = zeros(1,N);
        %Ƶ����չ
        if mod(N,2)==0      
            for i = 0:N/2
                mySpectrum(i+1) = TimeSum(i+1);
            end
            for i = N/2+1:N-1
                mySpectrum(i+1) = conj(TimeSum(N-i+1));
            end
        else
            for i = 0:(N-1)/2
                mySpectrum(i+1) = TimeSum(i+1);
            end
            for i = (N+1)/2:N-1
                mySpectrum(i+1) = conj(TimeSum(N-i+1));
            end
        end
%%%%%%%%%%%%%����STFT%%%%%%%%%%%%%%%%
        if (strcmp(tf_type, 'STFT'))
            gf = @(Omega) sqrt(2*s)*pi^(1/4)*exp(-(s*Omega).^2/2);
            g0 = conj(gf(0));
            Reconstruction = ifft(mySpectrum)/g0*dt;
%%%%%%%%%%%%%����MRM%%%%%%%%%%%%%%%%
        elseif (strcmp(tf_type, 'MRM'))
            gt = @(t) s^(-1/2)*pi^(-1/4).*exp(-t.^2/s^2/2);
            g0 = conj(gt(0));
            Reconstruction = ifft(mySpectrum)/g0;
%%%%%%%%%%%%%����TSST1��TSST2%%%%%%%%%%%%%%%%
        elseif (strcmp(tf_type, 'TSST1')||strcmp(tf_type, 'TSST2'))
            gf = @(Omega) sqrt(2*s)*pi^(1/4)*exp(-(s*Omega).^2/2);
            g0 = conj(gf(0));
            Reconstruction = ifft(mySpectrum)/g0;
%%%%%%%%%%%%%����TSET1%%%%%%%%%%%%%%%%            
        elseif (strcmp(tf_type, 'TSET1'))
            gt = @(t) s^(-1/2)*pi^(-1/4).*exp(-t.^2/s^2/2);
            g0 = conj(gt(0));
            Reconstruction = ifft(mySpectrum)/g0;
%%%%%%%%%%%%%����TSET2%%%%%%%%%%%%%%%%            
        elseif (strcmp(tf_type, 'TSET2')||strcmp(tf_type, 'DET'))
            Reconstruction = ifft(mySpectrum);
%%%%%%%%%%%%%����%%%%%%%%%%%%%%%%
        else
            error([direction,'��֧�֣�',tf_type]);
        end        
%% Ƶ�ʷ���ѹ���ı任�ع�      
    else
%%%%%%%%%%%%%���㹫������%%%%%%%%%%%%%%%% 
        mySpectrum = zeros(N,N);
        %Ƶ����չ
        if mod(N,2)==0      
            for i = 0:N/2
                mySpectrum(:,i+1) = TFxi(:,i+1);
            end
            for i = N/2+1:N-1
                mySpectrum(:,i+1) = conj(TFxi(:,N-i+1));
            end
        else
            for i = 0:(N-1)/2
                mySpectrum(:,i+1) = TFxi(:,i+1);
            end
            for i = (N+1)/2:N-1
                mySpectrum(:,i+1) = conj(TFxi(:,N-i+1));
            end
        end
        FrequencySum = real(sum(mySpectrum,2));
%%%%%%%%%%%%%����STFT%%%%%%%%%%%%%%%%
        if (strcmp(tf_type, 'STFT'))
            gt = @(t) s^(-1/2)*pi^(-1/4).*exp(-t.^2/s^2/2);
            g0 = conj(gt(0));
            Reconstruction = FrequencySum/g0*df;
%%%%%%%%%%%%%����MRM%%%%%%%%%%%%%%%%
        elseif (strcmp(tf_type, 'MRM'))
            gf = @(Omega) sqrt(2*s)*pi^(1/4)*exp(-(s*Omega).^2/2);
            g0 = conj(gf(0));
            Reconstruction = FrequencySum/g0;
%%%%%%%%%%%%%����SST1��SST2%%%%%%%%%%%%%%%%
        elseif (strcmp(tf_type, 'SST1')||strcmp(tf_type, 'SST2'))
            gt = @(t) s^(-1/2)*pi^(-1/4).*exp(-t.^2/s^2/2);
            g0 = conj(gt(0));
            Reconstruction = FrequencySum/g0;
%%%%%%%%%%%%%����SET1%%%%%%%%%%%%%%%%            
        elseif (strcmp(tf_type, 'SET1'))
            gf = @(Omega) sqrt(2*s)*pi^(1/4)*exp(-(s*Omega).^2/2);
            g0 = conj(gf(0));
            Reconstruction = FrequencySum/g0;
%%%%%%%%%%%%%����SET2%%%%%%%%%%%%%%%%            
        elseif (strcmp(tf_type, 'SET2')||strcmp(tf_type, 'DET'))
            Reconstruction = FrequencySum;
%%%%%%%%%%%%%����%%%%%%%%%%%%%%%%
        else
            error([direction,'��֧�֣�',tf_type]);
        end  
    end
end