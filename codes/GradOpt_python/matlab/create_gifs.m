%% 1

set(0, 'DefaultLineLineWidth', 2);
figure(1)

array=flip_angle_changes_optimization;

for ii=1:size(array,1)
subplot(1,3,1), plot((1:size(array,2))*0+18); hold on; title(sprintf('Flipangle after iter. %d',ii)); xlabel('NEX'); ylabel('FA [°]');
subplot(1,3,1), plot(flip_angle_changes_optimization(ii,:)); hold off; grid; legend('Ernst (Mi=SS)','MRIzero (Mi=1)');
axis([0 Inf 0 70]);
subplot(1,3,2), plot(longitudinal_comp_ernst_sim); hold on; title('M_z'); xlabel('NEX');
subplot(1,3,2), plot(longitudinal_comp_optimization(ii,:)); hold off;grid;
axis([0 Inf 0 1]);
subplot(1,3,3), plot(transverse_comp_ernst_sim); hold on; title('M_T'); xlabel('NEX');
subplot(1,3,3), plot(transverse_comp_optimization(ii,:)); hold off;grid;
axis([0 Inf 0 0.3]);
set(gcf, 'Outerposition',[280         638        1019         376])
drawnow
      frame = getframe(1);

      im = frame2im(frame);

      [imind,cm] = rgb2ind(im,32);

      if ii == 1
          imwrite(imind,cm,'out.gif','gif', 'Loopcount',inf);
      elseif ii==size(array,1)
          imwrite(imind,cm,'out.gif','gif','WriteMode','append','DelayTime',3);
      else
          imwrite(imind,cm,'out.gif','gif','WriteMode','append','DelayTime',0.0005);
      end

end

set(0, 'DefaultLineLineWidth', 0.5);

%% 2

load('scanner_dict.mat')

set(0, 'DefaultLineLineWidth', 2);
figure(1)

array=flip_angle_changes_optimization;

for ii=1:size(array,1)
subplot(1,3,1), plot((1:size(array,2))*0+18); hold on; title(sprintf('Flipangle after iter. %d',ii)); xlabel('NEX'); ylabel('FA [°]');
subplot(1,3,1), plot(flip_angle_changes_optimization(ii,:)); hold off; grid; legend('Ernst (Mi=SS)','MRIzero (Mi=1)');
axis([0 Inf 0 70]);
subplot(1,3,2), plot(longitudinal_comp_ernst_sim); hold on; title('M_z'); xlabel('NEX');
subplot(1,3,2), plot(longitudinal_comp_optimization(ii,:)); hold off;grid;
axis([0 Inf 0 1]);
subplot(1,3,3), plot(transverse_comp_ernst_sim); hold on; title('M_T'); xlabel('NEX');
subplot(1,3,3), plot(transverse_comp_optimization(ii,:)); hold off;grid;
axis([0 Inf 0 0.3]);
set(gcf, 'Outerposition',[280         638        1019         376])
drawnow
      frame = getframe(1);

      im = frame2im(frame);

      [imind,cm] = rgb2ind(im,32);

      if ii == 1
          imwrite(imind,cm,'out.gif','gif', 'Loopcount',inf);
      elseif ii==size(array,1)
          imwrite(imind,cm,'out.gif','gif','WriteMode','append','DelayTime',3);
      else
          imwrite(imind,cm,'out.gif','gif','WriteMode','append','DelayTime',0.0005);
      end

end

set(0, 'DefaultLineLineWidth', 0.5);
