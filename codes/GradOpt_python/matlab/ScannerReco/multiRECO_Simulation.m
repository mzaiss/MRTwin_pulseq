
clear all; close all;
%%
if isunix
  mrizero_git_dir = '/is/ei/aloktyus/git/mrizero_tueb';
  seq_dir = [mrizero_git_dir '/codes/GradOpt_python/out'];
else
  mrizero_git_dir = 'D:/root/ZAISS_LABLOG/LOG_MPI/27_MRI_zero/mrizero_tueb';
  seq_dir = 'D:\root\ZAISS_LABLOG\LOG_MPI\27_MRI_zero\mrizero_tueb\codes\GradOpt_python\out';
end

addpath([ mrizero_git_dir,'/codes/SequenceSIM']);
addpath([ mrizero_git_dir,'/codes/SequenceSIM/3rdParty/pulseq-master/matlab/']);

% experiment_id = 'FLASH_spoiled_lowSAR64_1kspins_multistep';
experiment_id = 'FLASH_spoiled_lowSAR32_multistep_190328';
%experiment_id = 'FLASH_spoiled_lowSAR_multistep';

% seq_dir = 'K:\CEST_seq\pulseq_zero\sequences';
experiment_id = 'GRE_LOWSAR_FA5_opt_grads_from_init_32_1knspins';
experiment_id = 'GRE_LOWSAR_FA20_optall_initgradmomstozero_1knspins';
% experiment_id = 'GRE_LOWSAR_FA20_optall_perturbgradmoms_1knspins';
experiment_id = 'FLASH_spoiled_lowSAR78_100spins';
experiment_id = 'FLASH_spoiled_lowSAR64_optgrads_frominit_1kspins_multistep';
experiment_id = 'FLASH_spoiled_lowSAR64_optgradmoms_1kspins_multistep';
experiment_id = 'FLASH_spoiled_lowSAR64_500spins_multistep';
experiment_id = 'FLASH_spoiled_lowSAR64_400spins_multistep';
experiment_id = 'tgtGRE_tsk_GRE_no_grad';


ni = 30;

scanner_dict = load([seq_dir,'/',experiment_id,'/','all_iter.mat']);
sz = double(scanner_dict.sz);
T = scanner_dict.T;
NRep = scanner_dict.NRep;

niter = size(scanner_dict.flips,1);
k = 1;
idxarray = [1:10,20:10:840];
array = 1:niter;
% array = [1:30,40:10:840];
    
sos_base= abs(squeeze(scanner_dict.reco_images(1,:,:,1)+1j*scanner_dict.reco_images(1,:,:,2)));
phase_base = angle(squeeze(scanner_dict.reco_images(1,:,:,1)+1j*scanner_dict.reco_images(1,:,:,2)));
SAR_base = sum(reshape((scanner_dict.flips(1,:,:,1).^2),1,[]));

sos_end= abs(squeeze(scanner_dict.reco_images(end,:,:,1)+1j*scanner_dict.reco_images(end,:,:,2)));
phase_end = angle(squeeze(scanner_dict.reco_images(end,:,:,1)+1j*scanner_dict.reco_images(end,:,:,2)));

n=0;
for ii=array
n=n+1; 
sos = abs(squeeze(scanner_dict.reco_images(ii,:,:,1)+1j*scanner_dict.reco_images(ii,:,:,2)));
phase = angle(squeeze(scanner_dict.reco_images(ii,:,:,1)+1j*scanner_dict.reco_images(ii,:,:,2)));
SAR = sum(reshape((squeeze(scanner_dict.flips(ii,:,:,1).^2)),1,[]))./SAR_base;

subplot(2,3,1), imagesc(flipud(flipud(sos_base)')); title(sprintf('SIM reco sos, iter %d',1)), axis('image'); colorbar;
ax=gca;
CLIM=ax.CLim;
CLIM=[-Inf Inf];
subplot(2,3,4), imagesc(flipud(flipud(phase_base)')), title('phase coil(1) '), axis('image'); colorbar;

subplot(2,3,3), imagesc(flipud(flipud(sos_end)')); title('sos, iter end'), axis('image'); colorbar;
ax=gca;
CLIM=ax.CLim;
CLIM=[-Inf Inf];
subplot(2,3,6), imagesc(flipud(flipud(phase_end)')), title(' phase coil(1) '), axis('image'); colorbar;

subplot(2,3,2), imagesc(flipud(flipud(sos)'),CLIM), title(sprintf(' sos, iter %d, SAR %.1f',n,SAR)), axis('image'); colorbar;
subplot(2,3,5), imagesc(flipud(flipud(phase)')), title(' phase coil(1) '), axis('image'); colorbar;
% set(gcf, 'Outerposition',[404   356   850   592]) %large
set(gcf, 'Outerposition',[451   346   598   398]) % small


% create gif (out.gif)
drawnow
      frame = getframe(1);
      im = frame2im(frame);
      im_SIM(:,:,:,n)=im;
      [imind,cm] = rgb2ind(im,32);
      if ii == 1
          imwrite(imind,cm,'out_sim.gif','gif', 'Loopcount',inf);
      elseif ii==numel(array)
          imwrite(imind,cm,'out_sim.gif','gif','WriteMode','append','DelayTime',0.2);
      else
          imwrite(imind,cm,'out_sim.gif','gif','WriteMode','append','DelayTime',0.0005);
      end
end
set(0, 'DefaultLineLineWidth', 0.5);

% save('togif.mat','im_SIM','experiment_id','-append');
