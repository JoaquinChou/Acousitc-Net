% size_le_marker = 20;
% reso = get(0, 'screensize');
% f = figure('Visible', 'on', 'Position', ...
%            [floor(reso(3)/2)-500, floor(reso(4)/2)-250, 1000, 500], ...
%            'Resize', 'off');
%        
% MicsPositionFigure = axes;
% title(MicsPositionFigure, 'Microphones');
% hold(MicsPositionFigure);
% set(MicsPositionFigure, ...
%     'Box', 'on', 'XGrid', 'on', 'YGrid', 'on', ...
%     'Position', [0.075 0.15 0.35 0.7], ...
%     'XTick', mic_x(1):mic_x(2), ...
%     'YTick', mic_y(1):mic_y(2), ...
%     'XLim', mic_x, 'YLim', mic_y);
% 
% plot(MicsPositionFigure, xcfg, ycfg, 'k.', 'MarkerSize', size_le_marker);
% 
% BeamformFigure = axes;
% hold(BeamformFigure);
% set(BeamformFigure, ...
%     'Box', 'on', 'XGrid', 'on', 'YGrid', 'on', ...
%     'Position', [0.525 0.15 0.35 0.7], ...
%     'XTick', scan_x(1):scan_x(2), ...
%     'YTick', scan_y(1):scan_y(2), ...
%     'XLim', scan_x, 'YLim', scan_y);



figure;
% set(gcf,'Position',[1000 1000 260 220]);
BeamformFigure = axes;
maxSPL = ceil(max(SPL(:)));
contourf(X, Y, SPL, [(maxSPL-BF_dr):1:maxSPL], 'Parent', BeamformFigure,'LineStyle','none');  %imagesc(X3,Z3,B3Lz);  
hold on; 
plot(x_Real, y_Real, 'p','MarkerSize',7,'color','k');

xlim([-1.5 1.5]);
ylim([-1.5 1.5])
title(['DAMAS'],'fontname','Times New Roman','fontsize',18,'FontWeight','bold');
xlabel('x (m)','fontsize',18,'fontname','Times New Roman','FontWeight','bold')
ylabel('y (m)','fontsize',18,'fontname','Times New Roman','FontWeight','bold')
colormap('hot');
cb = colorbar('peer', BeamformFigure, 'YTick', [(maxSPL-BF_dr):2:maxSPL]);
title(cb, 'SPL [dB]','fontsize',18,'fontname','Times New Roman','FontWeight','bold');
caxis([(maxSPL-BF_dr) maxSPL]);

% 计算瑞利熵
% SPL(SPL<=-10)=0;
% r = renyi(SPL,X,Y',3);
% text(-2.9,-2.7,['Renyi Entropy: ',num2str(r)],'Color','k','FontWeight','bold') 