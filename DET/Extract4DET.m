function [ExtractTFR,RestTFR] = Extract4DET(TFx,s,direction,delta,Rep,q,tradeoff)
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
%% Ԥ����
    [N,L] = size(TFx);
    ExtractTFR = zeros(N,L);
    TempTFR = zeros(N,L);
    lambda = tradeoff;
%% ʱ�䷽��ѹ���ı任�ع�
    if (strcmp(direction, 'Time') || strcmp(direction, 'T'))
%%%%%%%%%%%%%���㹫������%%%%%%%%%%%%%%%%
        [Cs,~] = brevridge(abs(TFx), 0:N-1, lambda);         
        for ptr =1:L
            minvalue = max(Cs(ptr)-round(delta/2),1);
            maxvalue = min(Cs(ptr)+round(delta/2),N);
            TempTFR(minvalue:maxvalue,ptr) = TFx(minvalue:maxvalue,ptr);
            for ptr2 =minvalue:maxvalue
                M0 = sqrt(-q(ptr2,ptr)+s^2);
                N0 = exp(-0.5*(-Rep(ptr2,ptr)./M0).^2);
                ExtractTFR(ptr2,ptr) = TempTFR(ptr2,ptr)*pi^(1/4)/sqrt(s)*M0*N0;                
            end
        end
        RestTFR = TFx - TempTFR;  
%% Ƶ�ʷ���ѹ���ı任�ع�       
    else
%%%%%%%%%%%%%���㹫������%%%%%%%%%%%%%%%%
        [Cs,~] = brevridge(abs(TFx'), 0:L-1, lambda);
        for ptr =1:N
            minvalue = max(Cs(ptr)-round(delta/2),1);
            maxvalue = min(Cs(ptr)+round(delta/2),L);
            TempTFR(ptr,minvalue:maxvalue) = TFx(ptr,minvalue:maxvalue);
            for ptr2 =minvalue:maxvalue
                M0 = sqrt(-q(ptr,ptr2)+1/(s^2));
                N0 = exp(-0.5*(-Rep(ptr,ptr2)./M0).^2);
                ExtractTFR(ptr,ptr2) = TempTFR(ptr,ptr2)/pi^(1/4)*sqrt(s)/sqrt(2)*M0*N0;              
            end
        end
        RestTFR = TFx - TempTFR;        
    end
end