n_source = length(source_x);

if n_source == 1
    % 单声源点
    bf_freq = 6000;  
elseif n_source == 2  
    % 双声源点
    bf_freq1 = 6000;  
    bf_freq2 = 8000;
    bf_freq = [bf_freq1 bf_freq2]';
elseif n_source == 3
    % 三声源点
    bf_freq1 = 6000;  
    bf_freq2 = 8000;
    bf_freq3 = 10000;
    bf_freq = [bf_freq1 bf_freq2 bf_freq3]';
elseif n_source == 4    
    % 四声源点
    bf_freq1 = 6000;  
    bf_freq2 = 8000;
    bf_freq3 = 10000;
    bf_freq4 = 12000;
    bf_freq = [bf_freq1 bf_freq2 bf_freq3 bf_freq4]';
end
