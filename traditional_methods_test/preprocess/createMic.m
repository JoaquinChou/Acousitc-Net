% mic_flag用于选择麦克风阵列 1~4
mic_flag = 1;

if mic_flag==1
    % 64点螺旋阵
    %初始点是0，螺距为0.01
    %在直角坐标系下b控制着螺线间距，b越大螺线间距越大
%     theta = zeros(1,64);
%     for i=1:64
%         theta(i) = 0.3*i*pi;
%     end
%     r = 0 + 0.01*theta;
%     xcfg = (r.*cos(theta))';
%     ycfg = (r.*sin(theta))';
    load 56_spiral_array.mat
    xcfg = array(:,1);
    ycfg = array(:,2);
    Mic_x_ac = mean(xcfg); Mic_y_ac = mean(ycfg); Mic_ac = [Mic_x_ac,Mic_y_ac,0];
elseif mic_flag==2
    % 64点方阵
    x = zeros(1,8);
    y = zeros(1,8);
    i = 1;
    for a = 1:8
        for b = 1:8
            x(i) = 1*(a-4.5);
            y(i) = 1*(b-4.5);
            i = i+1;
        end
    end    
    xcfg = x';
    ycfg = y';
    temp = max([xcfg;ycfg]);
    xcfg = xcfg./(temp+1)/10;
    ycfg = ycfg./(temp+1)/10;
elseif mic_flag==3
%     % 128点螺旋阵
%     theta = zeros(1,128);
%     for i=1:128
%         theta(i) = 0.3*i*pi;
%     end
%     r = 0 + 0.05*theta;
%     xcfg = (r.*cos(theta))';
%     ycfg = (r.*sin(theta))';
%     temp = max([xcfg;ycfg]);
%     xcfg = xcfg./(temp+1)/10;
%     ycfg = ycfg./(temp+1)/10;
    load tempx.mat
    load tempy.mat
    xcfg = tempx/100;
    ycfg = tempy/100;
elseif mic_flag==4
    % 128点方阵
    x = zeros(1,8);
    y = zeros(1,8);
    i = 1;
    for a = 1:16
        for b = 1:8
            x(i) = 1*(a-13.5);
            y(i) = 1*(b-4.5);
            i = i+1;
        end
    end    
    xcfg = x'-mean(x);
    ycfg = y'-mean(y);
    temp = max([xcfg;ycfg]);
    xcfg = xcfg./(temp+1)/10;
    ycfg = ycfg./(temp+1)/10;
end

