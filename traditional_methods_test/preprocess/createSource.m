% 构建声源点
source_flag = 1;
if source_flag==1
    % 一个声源点
    source_x = 0;
    source_y = 0;
elseif source_flag==2
    % 两个声源点
    source_x = [0 2]';
    source_y = [0 -2]';
elseif source_flag==3  
    % 三个声源点
    source_x = [0 2 -3]';
    source_y = [0 -2 3]';
elseif source_flag==4     
    % 四个声源点
    source_x = [0 2 3 -2]';
    source_y = [0 -2 -3 3]';
end
