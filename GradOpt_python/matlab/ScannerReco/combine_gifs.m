
load('togif.mat')

N=min(size(im_SIM,4),size(im_MEAS,4))
IMCOMB = cat(2,im_SIM(:,:,:,1:N),im_MEAS(:,:,:,1:N));
gifname=sprintf('out_simeas_%s.gif',experiment_id);
array=1:size(IMCOMB,4);
for ii=array
    im=IMCOMB(:,:,:,ii);
 [imind,cm] = rgb2ind(im,32);
      if ii == 1
          imwrite(imind,cm,gifname,'gif', 'Loopcount',inf);
      elseif ii==numel(array)
          imwrite(imind,cm,gifname,'gif','WriteMode','append','DelayTime',7);
      else
          imwrite(imind,cm,gifname,'gif','WriteMode','append','DelayTime',0.5);
      end
end;