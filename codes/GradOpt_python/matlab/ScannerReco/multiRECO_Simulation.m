
clear all; close all;

if isunix
  mrizero_git_dir = '/is/ei/aloktyus/git/mrizero_tueb';
  seq_dir = [mrizero_git_dir '/codes/GradOpt_python/out'];
else
  mrizero_git_dir = 'D:/root/ZAISS_LABLOG/LOG_MPI/27_MRI_zero/mrizero_tueb';
  seq_dir = 'K:\CEST_seq\pulseq_zero\sequences';
end

addpath([ mrizero_git_dir,'/codes/SequenceSIM']);
addpath([ mrizero_git_dir,'/codes/SequenceSIM/3rdParty/pulseq-master/matlab/']);

% experiment_id = 'FLASH_spoiled_lowSAR64_1kspins_multistep';
experiment_id = 'FLASH_spoiled_lowSAR32_multistep_190328';
%experiment_id = 'FLASH_spoiled_lowSAR_multistep';

seq_dir = 'K:\CEST_seq\pulseq_zero\sequences';
experiment_id = 'FLASH_spoiled_lowSAR64_400spins_multistep';

ni = 30;

scanner_dict = load([seq_dir,'/',experiment_id,'/','all_iter.mat']);
sz = double(scanner_dict.sz);
T = scanner_dict.T;
NRep = scanner_dict.NRep;

niter = size(scanner_dict.flips,1);
k = 1;
idxarray = [1:10,20:10:840];
array = 1:niter;
array = [1:30,40:10:840];
    
sos_base= abs(squeeze(scanner_dict.reco_images(1,:,:,1)+1j*scanner_dict.reco_images(1,:,:,2)));
phase_base = angle(squeeze(scanner_dict.reco_images(1,:,:,1)+1j*scanner_dict.reco_images(1,:,:,2)));
SAR_base = sum(reshape((scanner_dict.flips(1,:,:,1).^2),1,[]));

for ii=array
    
sos = abs(squeeze(scanner_dict.reco_images(ii,:,:,1)+1j*scanner_dict.reco_images(ii,:,:,2)));
phase = angle(squeeze(scanner_dict.reco_images(ii,:,:,1)+1j*scanner_dict.reco_images(ii,:,:,2)));
SAR = sum(reshape((squeeze(scanner_dict.flips(ii,:,:,1).^2)),1,[]))./SAR_base;

subplot(2,2,1), imagesc(flipud(flipud(sos)')), title(sprintf('reco sos, iter %d, SAR %f',ii,SAR)), axis('image'); colorbar;
subplot(2,2,2), imagesc(flipud(flipud(phase)')), title('reco phase coil(1) '), axis('image'); colorbar;
subplot(2,2,3), imagesc(flipud(flipud(sos_base)')), title(sprintf('reco sos, iter %d',1)), axis('image'); colorbar;
subplot(2,2,4), imagesc(flipud(flipud(phase_base)')), title('reco phase coil(1) '), axis('image'); colorbar;
set(gcf, 'Outerposition',[404   356   850   592])

% create gif (out.gif)
drawnow
      frame = getframe(1);
      im = frame2im(frame);
      [imind,cm] = rgb2ind(im,32);
      if ii == 1
          imwrite(imind,cm,'out_sim.gif','gif', 'Loopcount',inf);
      elseif ii==numel(array)
          imwrite(imind,cm,'out_sim.gif','gif','WriteMode','append','DelayTime',3);
      else
          imwrite(imind,cm,'out_sim.gif','gif','WriteMode','append','DelayTime',0.0005);
      end
end
set(0, 'DefaultLineLineWidth', 0.5);


