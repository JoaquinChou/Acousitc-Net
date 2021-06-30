function [ExtractTFR,RestTFR] = Extract4DET(TFx,s,direction,delta,Rep,q,tradeoff)
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
%% 预处理
    [N,L] = size(TFx);
    ExtractTFR = zeros(N,L);
    TempTFR = zeros(N,L);
    lambda = tradeoff;
%% 时间方向压缩的变换重构
    if (strcmp(direction, 'Time') || strcmp(direction, 'T'))
%%%%%%%%%%%%%计算公共部分%%%%%%%%%%%%%%%%
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
%% 频率方向压缩的变换重构       
    else
%%%%%%%%%%%%%计算公共部分%%%%%%%%%%%%%%%%
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