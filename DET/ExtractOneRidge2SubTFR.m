function [ExtractTFR,RestTFR] = ExtractOneRidge2SubTFR(TFx,fs,s,direction,tf_type,delta_or_Rep,q,tradeoff)
%% 参数说明
%函数名：提取单条脊线得到子时频表示；英文名：Extract One Ridge to SubTFR
%%%%%%%%%%%%%%输入参数%%%%%%%%%%%%%%
%TFx:待分析时频表示
%fs:采样率
%s:窗长度
%direction:脊线提取方向
%tf_type:时频方法类型
%delta_or_Rep:重构参数，在STFT、压缩类方法中表示脊线重构的范围，物理量位时间宽度或频率宽度；在二阶提取类变换中表示一个重构因子
%q:二阶提取类变换中重构因子
%tradeoff：脊线提取参数，brevridge函数用到
%%%%%%%%%%%%%%输出参数%%%%%%%%%%%%%%
%ExtractTFR：所提取的子TFR
%RestTFR：剩下的的TFR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 参数检查
    if (nargin > 8)
        error('输入参数过多！');
    elseif(nargin == 7)
        tradeoff = 0.009;  
    elseif(nargin == 6 || nargin == 5 || nargin == 4 || nargin == 3 || nargin == 2 || nargin == 1 || nargin == 0)
        error('缺少输入参数！');
    end
%% 预处理
    [N,L] = size(TFx);
    ExtractTFR = zeros(N,L);
    lambda = tradeoff;
%% 时间方向压缩的变换重构
    if (strcmp(direction, 'Time') || strcmp(direction, 'T'))
%%%%%%%%%%%%%计算公共部分%%%%%%%%%%%%%%%%
        [Cs,~] = brevridge(abs(TFx), 0:N-1, lambda);
%%%%%%%%%%%%%计算STFT和TSST1和TSST2%%%%%%%%%%%%%%%%
        if (strcmp(tf_type, 'STFT')||strcmp(tf_type, 'TSST1')||strcmp(tf_type, 'TSST2'))
            delta = delta_or_Rep*fs;
            for ptr =1:L
                minvalue = max(Cs(ptr)-round(delta/2),1);
                maxvalue = min(Cs(ptr)+round(delta/2),N);
                ExtractTFR(minvalue:maxvalue,ptr) = TFx(minvalue:maxvalue,ptr);
            end
            RestTFR = TFx - ExtractTFR;
%%%%%%%%%%%%%计算TSET1%%%%%%%%%%%%%%%%            
        elseif (strcmp(tf_type, 'TSET1')||strcmp(tf_type, 'MRM'))
            for ptr =1:L
                ExtractTFR(Cs(ptr),ptr) = TFx(Cs(ptr),ptr);
            end
            RestTFR = TFx - ExtractTFR;
%%%%%%%%%%%%%计算TSET2%%%%%%%%%%%%%%%%            
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
%%%%%%%%%%%%%其它%%%%%%%%%%%%%%%%
        else
            error([direction,'不支持：',tf_type]);
        end        
%% 频率方向压缩的变换重构       
    else
%%%%%%%%%%%%%计算公共部分%%%%%%%%%%%%%%%%
        [Cs,~] = brevridge(abs(TFx'), 0:L-1, lambda);
%%%%%%%%%%%%%计算STFT和SST1和SST2%%%%%%%%%%%%%%%%
        if (strcmp(tf_type, 'STFT')||strcmp(tf_type, 'SST1')||strcmp(tf_type, 'SST2'))
            delta = delta_or_Rep/fs*N;
            for ptr =1:N
                minvalue = max(Cs(ptr)-round(delta/2),1);
                maxvalue = min(Cs(ptr)+round(delta/2),L);
                ExtractTFR(ptr,minvalue:maxvalue) = TFx(ptr,minvalue:maxvalue);
            end
            RestTFR = TFx - ExtractTFR;
%%%%%%%%%%%%%计算SET1%%%%%%%%%%%%%%%%            
        elseif (strcmp(tf_type, 'SET1')||strcmp(tf_type, 'MRM'))
            for ptr =1:N
                ExtractTFR(ptr,Cs(ptr)) = TFx(ptr,Cs(ptr));
            end
            RestTFR = TFx - ExtractTFR;
%%%%%%%%%%%%%计算SET2%%%%%%%%%%%%%%%%            
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
%%%%%%%%%%%%%其它%%%%%%%%%%%%%%%%
        else
            error([direction,'不支持：',tf_type]);
        end          
    end
end