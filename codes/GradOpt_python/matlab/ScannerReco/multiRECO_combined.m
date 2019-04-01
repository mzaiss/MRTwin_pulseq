function multiRECO_combined
%% very simple FFT reconstruction from raw data
addpath('D:\root\ZAISS_LABLOG\LOG_MPI\0_CESTtool\CEST_EVAL_GLINT\reco_raw\recoVBVD')
addpath('D:\root\ZAISS_LABLOG\LOG_MPI\0_CESTtool\CEST_EVAL_GLINT\reco_raw\mapVBVD')
%% Load raw data
%MEAS
origpath=pwd;
d = uigetdir('\\mrz3t\Upload\CEST_seq\pulseq_zero\sequences', 'Select a sequence folder');
out=regexp(d,'\','split');
experiment_id=out{end};
files = dir(fullfile(d, '/data/*.dat'));
array_MEAS=1:numel(files);

twix_obj = mapVBVD([d '/data/' files(1).name]);
[sos_base, phase_base] = TWIXtoIMG_FFT(twix_obj);

%SIM
scanner_dict = load([d,'/','all_iter.mat']);
sz = double(scanner_dict.sz);
T = scanner_dict.T;
NRep = scanner_dict.NRep;

niter = size(scanner_dict.flips,1);
array_SIM = [1:30,40:10:840];
array_SIM=array_SIM(1:niter);

SIM_sos_base= abs(squeeze(scanner_dict.reco_images(1,:,:,1)+1j*scanner_dict.reco_images(1,:,:,2)));
SIM_phase_base = angle(squeeze(scanner_dict.reco_images(1,:,:,1)+1j*scanner_dict.reco_images(1,:,:,2)));
SIM_SAR_base = sum(reshape((scanner_dict.flips(1,:,:,1).^2),1,[]));

figure(1);  set(0, 'DefaultLineLineWidth', 2);  % prep for gif
for ii=array_MEAS
    
%MEAS
    filename=files(ii).name;
twix_obj = mapVBVD([d '/data/' filename]);
[sos, phase] = TWIXtoIMG_FFT(twix_obj);

subplot(3,4,3), imagesc(rot90(sos),[0 1]), title(sprintf('reco sos, iter %d',ii)), axis('image'); colorbar; set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);
subplot(3,4,7), imagesc(rot90(phase)), title('reco phase coil(1) '), axis('image'); colorbar; set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);
subplot(3,4,4), imagesc(rot90(sos_base),[0 1]), title(sprintf('MEAS reco sos, iter %d',1)), axis('image'); colorbar; set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);
subplot(3,4,8), imagesc(rot90(phase_base)), title('reco phase coil(1) '), axis('image'); colorbar; set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);

%SIM
sos = abs(squeeze(scanner_dict.reco_images(ii,:,:,1)+1j*scanner_dict.reco_images(ii,:,:,2)));
phase = angle(squeeze(scanner_dict.reco_images(ii,:,:,1)+1j*scanner_dict.reco_images(ii,:,:,2)));
SAR = sum(reshape((squeeze(scanner_dict.flips(ii,:,:,1).^2)),1,[]))./SIM_SAR_base;
subplot(3,4,1), imagesc(flipud(flipud(SIM_sos_base)')); title(sprintf('SIM reco sos, iter %d',1)), axis('image'); colorbar; set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);
ax=gca;
CLIM=ax.CLim;
CLIM=[-Inf Inf];
subplot(3,4,5), imagesc(flipud(flipud(SIM_phase_base)')), title('reco phase coil(1) '), axis('image'); colorbar; set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);
subplot(3,4,2), imagesc(flipud(flipud(sos)'),CLIM), title(sprintf('reco sos, iter %d, SAR %f',ii,SAR)), axis('image'); colorbar; set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);
subplot(3,4,6), imagesc(flipud(flipud(phase)')), title('reco phase coil(1) '), axis('image'); colorbar;set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);
set(gcf, 'Outerposition',[119          69        1444         897])

% colormap 'jet'
subplot(6,3,13), imagesc(squeeze(scanner_dict.flips(ii,:,:,1))'); title('Flips'); colorbar; set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);
subplot(6,3,16), imagesc(squeeze(scanner_dict.flips(ii,:,:,2))'); title('Phases');colorbar; set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);
subplot(6,3,14), imagesc(squeeze(scanner_dict.event_times(ii,:,:))'); title('delays');colorbar; set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);
subplot(6,3,15), imagesc(squeeze(scanner_dict.grad_moms(ii,:,:,1))');         title('gradmomx');colorbar; set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);
subplot(6,3,18), imagesc(squeeze(scanner_dict.grad_moms(ii,:,:,2))');          title('gradmomy');colorbar; set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);

% create gif (out.gif)
drawnow
      frame = getframe(1);
      im = frame2im(frame);
      im_MEAS(:,:,:,ii)=im;
      [imind,cm] = rgb2ind(im,32);
      gifname=sprintf('TEST_out_simeas_%s.gif',experiment_id);
      if ii == 1
          imwrite(imind,cm,gifname,'gif', 'Loopcount',inf);
      elseif ii==numel(array_MEAS)
          imwrite(imind,cm,gifname,'gif','WriteMode','append','DelayTime',3);
      else
          imwrite(imind,cm,gifname,'gif','WriteMode','append','DelayTime',0.0005);
      end
end
set(0, 'DefaultLineLineWidth', 0.5); % prep for gif

end


function [SOS, phase] = TWIXtoIMG_FFT(twix_obj)
%% sort in the k-space data
if iscell(twix_obj)
    data_unsorted = twix_obj{2}.image.unsorted();
else
    data_unsorted = twix_obj.image.unsorted();
end
% the incoming data order is [kx coils acquisitions]
data_coils_last = permute(data_unsorted, [1, 3, 2]);
nCoils = size(data_coils_last, 3);

data = data_coils_last;

%% Reconstruct coil images
images = zeros(size(data));
%figure;

for ii = 1:nCoils
    %images(:,:,:,ii) = fliplr(rot90(fftshift(fft2(fftshift(data(:,:,:,ii))))));
    images(:,:,ii) = fftshift(fft2(fftshift(data(end:-1:1,:,ii))));
end

% Phase images (possibly channel-by-channel and echo-by-echo)
% figure;
% imab(angle(images));colormap('jet');

sos=abs(sum(images.^2,ndims(images)).^(1/2));
SOS=sos./max(sos(:));
phase = angle(images(:,:,ii));

if 0
%% Sum of squares combination of channels
figure(101), subplot(2,2,3)
imagesc(rot90(sos)), title('reco coilcombination'), axis('image')

show kspace data
figure(101)
subplot(2,2,1), imagesc(rot90(abs(data(:,:,1)))), title('coil 1'), axis('image')
subplot(2,2,2), imagesc(rot90(abs(data(:,:,2)))), title('coil 2'), axis('image')
end

end