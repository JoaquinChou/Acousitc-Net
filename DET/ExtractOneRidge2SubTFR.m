function [ExtractTFR,RestTFR] = ExtractOneRidge2SubTFR(TFx,fs,s,direction,tf_type,delta_or_Rep,q,tradeoff)
%% ����˵��
%����������ȡ�������ߵõ���ʱƵ��ʾ��Ӣ������Extract One Ridge to SubTFR
%%%%%%%%%%%%%%�������%%%%%%%%%%%%%%
%TFx:������ʱƵ��ʾ
%fs:������
%s:������
%direction:������ȡ����
%tf_type:ʱƵ��������
%delta_or_Rep:�ع���������STFT��ѹ���෽���б�ʾ�����ع��ķ�Χ��������λʱ���Ȼ�Ƶ�ʿ�ȣ��ڶ�����ȡ��任�б�ʾһ���ع�����
%q:������ȡ��任���ع�����
%tradeoff��������ȡ������brevridge�����õ�
%%%%%%%%%%%%%%�������%%%%%%%%%%%%%%
%ExtractTFR������ȡ����TFR
%RestTFR��ʣ�µĵ�TFR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% �������
    if (nargin > 8)
        error('����������࣡');
    elseif(nargin == 7)
        tradeoff = 0.009;  
    elseif(nargin == 6 || nargin == 5 || nargin == 4 || nargin == 3 || nargin == 2 || nargin == 1 || nargin == 0)
        error('ȱ�����������');
    end
%% Ԥ����
    [N,L] = size(TFx);
    ExtractTFR = zeros(N,L);
    lambda = tradeoff;
%% ʱ�䷽��ѹ���ı任�ع�
    if (strcmp(direction, 'Time') || strcmp(direction, 'T'))
%%%%%%%%%%%%%���㹫������%%%%%%%%%%%%%%%%
        [Cs,~] = brevridge(abs(TFx), 0:N-1, lambda);
%%%%%%%%%%%%%����STFT��TSST1��TSST2%%%%%%%%%%%%%%%%
        if (strcmp(tf_type, 'STFT')||strcmp(tf_type, 'TSST1')||strcmp(tf_type, 'TSST2'))
            delta = delta_or_Rep*fs;
            for ptr =1:L
                minvalue = max(Cs(ptr)-round(delta/2),1);
                maxvalue = min(Cs(ptr)+round(delta/2),N);
                ExtractTFR(minvalue:maxvalue,ptr) = TFx(minvalue:maxvalue,ptr);
            end
            RestTFR = TFx - ExtractTFR;
%%%%%%%%%%%%%����TSET1%%%%%%%%%%%%%%%%            
        elseif (strcmp(tf_type, 'TSET1')||strcmp(tf_type, 'MRM'))
            for ptr =1:L
                ExtractTFR(Cs(ptr),ptr) = TFx(Cs(ptr),ptr);
            end
            RestTFR = TFx - ExtractTFR;
%%%%%%%%%%%%%����TSET2%%%%%%%%%%%%%%%%            
        elseif (strcmp(tf_type, 'TSET2')||strcmp(tf_type, 'DET'))
            Rep = delta_or_Rep;
            for ptr =1:L
                ExtractTFR(Cs(ptr),ptr) = TFx(Cs(ptr),ptr);
            end
            RestTFR = TFx - ExtractTFR;
            for ptr =1:L
                M0 = sqrt(-q(Cs(ptr),ptr)+s^2);
                N0 = exp(-0.5*(-Rep(Cs(ptr),ptr)./M0).^2);
                ExtractTFR(Cs(ptr),ptr) = ExtractTFR(Cs(ptr),ptr)*pi^(1/4)/sqrt(s)*M0*N0;
            end
%%%%%%%%%%%%%����%%%%%%%%%%%%%%%%
        else
            error([direction,'��֧�֣�',tf_type]);
        end        
%% Ƶ�ʷ���ѹ���ı任�ع�       
    else
%%%%%%%%%%%%%���㹫������%%%%%%%%%%%%%%%%
        [Cs,~] = brevridge(abs(TFx'), 0:L-1, lambda);
%%%%%%%%%%%%%����STFT��SST1��SST2%%%%%%%%%%%%%%%%
        if (strcmp(tf_type, 'STFT')||strcmp(tf_type, 'SST1')||strcmp(tf_type, 'SST2'))
            delta = delta_or_Rep/fs*N;
            for ptr =1:N
                minvalue = max(Cs(ptr)-round(delta/2),1);
                maxvalue = min(Cs(ptr)+round(delta/2),L);
                ExtractTFR(ptr,minvalue:maxvalue) = TFx(ptr,minvalue:maxvalue);
            end
            RestTFR = TFx - ExtractTFR;
%%%%%%%%%%%%%����SET1%%%%%%%%%%%%%%%%            
        elseif (strcmp(tf_type, 'SET1')||strcmp(tf_type, 'MRM'))
            for ptr =1:N
                ExtractTFR(ptr,Cs(ptr)) = TFx(ptr,Cs(ptr));
            end
            RestTFR = TFx - ExtractTFR;
%%%%%%%%%%%%%����SET2%%%%%%%%%%%%%%%%            
        elseif (strcmp(tf_type, 'SET2')||strcmp(tf_type, 'DET'))
            Rep = delta_or_Rep;
            for ptr =1:N
                ExtractTFR(ptr,Cs(ptr)) = TFx(ptr,Cs(ptr));
            end
            RestTFR = TFx - ExtractTFR;
            for ptr =1:N
                M0 = sqrt(-q(ptr,Cs(ptr))+1/(s^2));
                N0 = exp(-0.5*(-Rep(ptr,Cs(ptr))./M0).^2);
                ExtractTFR(ptr,Cs(ptr)) = ExtractTFR(ptr,Cs(ptr))/pi^(1/4)*sqrt(s)/sqrt(2)*M0*N0;
            end
%%%%%%%%%%%%%����%%%%%%%%%%%%%%%%
        else
            error([direction,'��֧�֣�',tf_type]);
        end          
    end
end