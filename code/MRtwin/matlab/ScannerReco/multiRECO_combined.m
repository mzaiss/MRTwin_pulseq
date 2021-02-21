function multiRECO_combined(single)


%% very simple FFT reconstruction from raw data
fclose('all');
addpath('D:\root\ZAISS_LABLOG\LOG_MPI\27_MRI_zero\mrizero_tueb\codes\SequenceSIM\3rdParty\pulseq-master')
addpath('D:\root\ZAISS_LABLOG\LOG_MPI\0_CESTtool\CEST_EVAL_GLINT\reco_raw\recoVBVD')
addpath('D:\root\ZAISS_LABLOG\LOG_MPI\0_CESTtool\CEST_EVAL_GLINT\reco_raw\mapVBVD')
%% Load raw data
%MEAS
origpath=pwd;

if isunix
    dir_seq = 'smb://mrz3t/upload/CEST_seq/pulseq_zero/sequences';
    d = '/media/upload3t/CEST_seq/pulseq_zero/sequences/FLASH_spoiled_lowSAR64_400spins_multistep';
    d = '/media/upload3t/CEST_seq/pulseq_zero/sequences/seq190506/t05_tgtEPI_tskEPI';
elseif ispc
    dir_seq = '/media/upload3t/CEST_seq/pulseq_zero/sequences';
    d = uigetdir('\\mrz3t\Upload\CEST_seq\pulseq_zero\sequences', 'Select a sequence folder');
else
    disp('Platform not supported')
end

%SIM
scanner_dict = load([d,'/','all_iter.mat']);
scanner_dict_tgt = load([d,'/','scanner_dict_tgt.mat']);
sz = double(scanner_dict.sz);
T = scanner_dict.T;
NRep = scanner_dict.NRep;

seq=mr.Sequence();
seq.read([d '/' sprintf('seqiter%d.seq',0)],'detectRFuse');
[ktraj_adc, ktraj] = seq.calculateKspace();
kmax=(max(ktraj_adc(1,1:sz(1)))-min(ktraj_adc(1,1:sz(1))))/2;

try
load([d,'/','export_protocol.mat'],'idxarray_exported_itersteps');
catch
    idxarray_exported_itersteps=1:10000;
end
array_SIM = idxarray_exported_itersteps;


SIM_sos_base= abs(squeeze(scanner_dict.reco_images(1,:,:,1)+1j*scanner_dict.reco_images(1,:,:,2)));
SIM_phase_base = angle(squeeze(scanner_dict.reco_images(1,:,:,1)+1j*scanner_dict.reco_images(1,:,:,2)));
SIM_SAR_base = sum(reshape((scanner_dict.flips(1,:,:,1).^2),1,[]));

SIM_sos_base= abs(squeeze(scanner_dict_tgt.reco(:,:,1)+1j*scanner_dict_tgt.reco(:,:,2)));
SIM_phase_base = angle(squeeze(scanner_dict_tgt.reco(:,:,1)+1j*scanner_dict_tgt.reco(:,:,2)));
SIM_SAR_base = sum(reshape((scanner_dict_tgt.flips(:,:,1).^2),1,[]));


%MEAS
out=regexp(d,'\','split');
experiment_id=out{end};
files = dir(fullfile(d, '/data/*.dat'));
array_MEAS=1:numel(files)-1;

twix_obj = mapVBVD([d '/data/' files(1).name]);

% [sos_base, phase_base] = TWIXtoIMG_NUFFT_gradmoms_from_pulseq(twix_obj,ktraj_adc);
%  [sos_base, phase_base] = TWIXtoIMG_ADJOINT(twix_obj, scanner_dict, 1);
   [sos_base, phase_base] = TWIXtoIMG_ADJOINT_gradmoms_from_pulseq(twix_obj,ktraj_adc);
%   [sos_base, phase_base] = TWIXtoIMG_FFT(twix_obj);

% save([d '/data/twix_obj_array.mat'],'twix_obj_array');
try
  if isunix
    load([d '/data/twix_obj_array_unix.mat']);
  else
    load([d '/data/twix_obj_array.mat']);
  end
catch
    for ii=array_MEAS
        filename=files(ii).name;
        twix_obj = mapVBVD([d '/data/' filename]);
        twix_obj_array{ii}=twix_obj;
    end
    if isunix
      save([d '/data/twix_obj_array_unix.mat'],'twix_obj_array');
    else
        save([d '/data/twix_obj_array.mat'],'twix_obj_array');
    end
end

if single>0
    array_MEAS = single; % only a single fram to display
end

figure(1);  
for ii=array_MEAS
    
%  seq.read([d '/' sprintf('seqiter%d.seq',ii)],'detectRFuse');
 
 seq.read([d '/' sprintf('seqiter%d.seq',ii)]);
try
[ktraj_adc, ktraj] = seq.calculateKspace();
catch
    warning(sprintf('use old k-space (%d for seqiter%d.seq)',ii-1,ii));
end

%MEAS
filename=files(ii).name;
twix_obj=twix_obj_array{ii};
% [sos, phase] = TWIXtoIMG_FFT(twix_obj);
% [sos, phase] = TWIXtoIMG_NUFFT_gradmoms_from_pulseq(twix_obj,ktraj_adc);
[sos, phase] = TWIXtoIMG_ADJOINT_gradmoms_from_pulseq(twix_obj,ktraj_adc);
% [sos, phase] = TWIXtoIMG_ADJOINT(twix_obj, scanner_dict, ii);

if 1 % 1= detailed plot
figure(1);  
subplot(3,4,3), imagesc(rot90(sos),[0 1]), title(sprintf('reco sos, seqiter %d',ii)), axis('image');  set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);
subplot(3,4,7), imagesc(rot90(phase)), title('reco phase coil(1) '), axis('image');  set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);
subplot(3,4,4), imagesc(rot90(sos_base),[0 1]), title(sprintf('MEAS reco sos, iter %d',1)), axis('image');  set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);
subplot(3,4,8), imagesc(rot90(phase_base)), title('reco phase coil(1) '), axis('image');  set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);

%SIM
jj=array_SIM(ii);
SIM_sos = abs(squeeze(scanner_dict.reco_images(jj,:,:,1)+1j*scanner_dict.reco_images(jj,:,:,2)));
SIM_phase = angle(squeeze(scanner_dict.reco_images(jj,:,:,1)+1j*scanner_dict.reco_images(jj,:,:,2)));
SIM_SAR = sum(reshape((squeeze(scanner_dict.flips(jj,:,:,1).^2)),1,[]))./SIM_SAR_base;
subplot(3,4,1), imagesc(rot90(SIM_sos_base.',2)); title(sprintf('SIM reco sos, iter %d',1)), axis('image');  set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);
ax=gca;
CLIM=ax.CLim;
CLIM=[-Inf Inf];

% [SIM_sos, SIM_phase] = SIMtoIMG_NUFFT(scanner_dict,ktraj_adc,ii);  % overwrites generated targets

subplot(3,4,5), imagesc(rot90((SIM_phase_base).',2)), title('reco phase coil(1) '), axis('image');  set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);
subplot(3,4,2), imagesc(rot90((SIM_sos).',2),CLIM), title(sprintf('reco sos, optiter %d, SAR %f',jj,SIM_SAR)), axis('image');  set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);
subplot(3,4,6), imagesc(rot90((SIM_phase).',2)), title('reco phase coil(1) '), axis('image'); set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);

if ispc
  set(gcf, 'Outerposition',[10          82        1362         883])
end



% subplot(6,4,18), imagesc(squeeze(scanner_dict.flips(jj,:,:,1))'); title('Flips'); colorbar; set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);
% subplot(6,4,22), imagesc(squeeze(scanner_dict.flips(jj,:,:,2))'); title('Phases');colorbar; set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);

% subplot(6,4,18), plot(180/pi*squeeze(scanner_dict_tgt.flips(1,:,1)),'g','DisplayName','flips tgt'); hold on;% a 2D plot
% subplot(6,4,18), plot(180/pi*abs(squeeze(scanner_dict.flips(jj,1,:,1))),'r','DisplayName','flips'); 
% subplot(6,4,18), plot(180/pi*squeeze(scanner_dict.flips(jj,1,:,1)),'r--','DisplayName','flips'); hold off; axis([-Inf Inf 0 Inf]);

subplot(6,4,18), plot(180/pi*squeeze(scanner_dict_tgt.flips(1,:,1)),'r.','DisplayName','flips tgt'); hold on;% a 2D plot
subplot(6,4,18), plot(180/pi*(squeeze(scanner_dict.flips(ii,1,:,1))),'r','DisplayName','flips'); 
subplot(6,4,18), plot(180/pi*squeeze(scanner_dict_tgt.flips(2,:,1)),'b.','DisplayName','flips tgt'); % a 2D plot
subplot(6,4,18), plot(180/pi*(squeeze(scanner_dict.flips(ii,2,:,1))),'b','DisplayName','flips');hold off;  xlabel('rep'); ylabel('flip angle [°]');

subplot(6,4,22), plot(180/pi*squeeze(scanner_dict.flips(ii,1,:,2)),'r','DisplayName','phase1'); hold on;% a 2D plot
subplot(6,4,22), plot(180/pi*squeeze(scanner_dict.flips(ii,2,:,2)),'b','DisplayName','phase2'); hold off; xlabel('rep'); ylabel('phase angle [°]');

subplot(6,4,19), imagesc(squeeze(scanner_dict.event_times(jj,:,:))'); title('delays');colorbar; set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);
subplot(6,4,20), imagesc(squeeze(scanner_dict.grad_moms(jj,:,:,1))');         title('gradmomx');colorbar; set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);
subplot(6,4,24), imagesc(squeeze(scanner_dict.grad_moms(jj,:,:,2))');          title('gradmomy');colorbar; set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);


% plot k-spaces
subplot(3,4,9), 
plot(ktraj(1,:),ktraj(2,:),'b'); hold on; % a 2D plot 
% axis('equal'); % enforce aspect ratio for the correct trajectory display
plot(ktraj_adc(1,:),ktraj_adc(2,:),'r.'); axis('equal'); title('k-space samples')
grid on; hold off; axis('equal');
set(gca,'XTick',[-kmax kmax]); set(gca,'YTick',[-kmax kmax]); set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);

pause(0.1);

write_gif(ii,numel(array_MEAS),1,d,experiment_id,'full')
end


if 0 %  small plots, sos + kspace
    figure(2);  
    jj=array_SIM(ii);
    seq.read([d '/' sprintf('seqiter%d.seq',ii)],'detectRFuse');
   [ktraj_adc, ktraj] = seq.calculateKspace();
    subplot(2,1,1), imagesc(rot90(sos),[0 1]), title(sprintf('reco sos, iter %d',jj)), axis('image'); colorbar; set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);
    subplot(2,1,2), 

% % RF
% plot(180/pi*squeeze(scanner_dict_tgt.flips(1,:,1)),'g','DisplayName','flips tgt'); hold on;% a 2D plot
% plot(180/pi*abs(squeeze(scanner_dict.flips(jj,1,:,1))),'r','DisplayName','flips'); 
% plot(180/pi*squeeze(scanner_dict.flips(jj,1,:,1)),'r--','DisplayName','flips'); hold off; axis([-Inf Inf 0 Inf]);

% % grad
plot(ktraj(1,:),ktraj(2,:),'b'); hold on; % a 2D plot 
% axis('equal'); % enforce aspect ratio for the correct trajectory display
plot(ktraj_adc(1,:),ktraj_adc(2,:),'r.'); axis('equal'); title('k-space samples')
grid on; hold off; 
set(gca,'XTick',[-kmax kmax]); set(gca,'YTick',[-kmax kmax]); set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);
% pause(0.1);
    set(gcf, 'Outerposition',[1362         272         294         480])
    if (single==0)
        write_gif(ii,numel(array_MEAS),2,d,experiment_id,'small')
    end
end

end


end

function write_gif(ii,max_ii,fh,path,experiment_id,suffix)
if ispc
    drawnow
    frame = getframe(fh);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,32);
    
    % write gif only if windows
    gifname=sprintf('%s/%s_%s.gif',path,experiment_id,suffix);
    if ii == 1
        imwrite(imind,cm,gifname,'gif', 'Loopcount',inf);
    elseif ii==max_ii
        imwrite(imind,cm,gifname,'gif','WriteMode','append','DelayTime',5);
    elseif ii<10
        imwrite(imind,cm,gifname,'gif','WriteMode','append','DelayTime',0.3);
    else
        imwrite(imind,cm,gifname,'gif','WriteMode','append','DelayTime',0.0005);
    end
    
end
end
function [G_adj] = get_adjoint_mtx(grad_moms,adc_mask,T,NRep,NVox,sz)

  G_adj = zeros(T,NRep,NVox,3,3);
  G_adj(:,:,:,3,3) = 1;
  
  grad_moms = padarray(grad_moms,[1,0,0],'pre');
  grad_moms = grad_moms(1:end-1,:,:);    

  k = cumsum(grad_moms,1);

  % get ramps
  baserampX = linspace(-1,1,sz + 1);
  baserampY = linspace(-1,1,sz + 1);

  rampX = pi*baserampX;
  rampX = -rampX(1:sz).'*ones(1,sz);
  rampX = reshape(rampX,[1,1,NVox]);

  rampY = pi*baserampY;
  rampY = -ones(sz,1)*rampY(1:sz);
  rampY = reshape(rampY,[1,1,NVox]);

  B0X = reshape(k(:,:,1), [T,NRep,1]) .* rampX;
  B0Y = reshape(k(:,:,2), [T,NRep,1]) .* rampY;

  B0_grad = reshape((B0X + B0Y), [T,NRep,NVox]);

  B0_grad_adj_cos = cos(B0_grad);
  B0_grad_adj_sin = sin(B0_grad);      

  % adjoint
  G_adj(:,:,:,1,1) = B0_grad_adj_cos;
  G_adj(:,:,:,1,2) = B0_grad_adj_sin;
  G_adj(:,:,:,2,1) = -B0_grad_adj_sin;
  G_adj(:,:,:,2,2) = B0_grad_adj_cos;

  G_adj = permute(G_adj,[3,4,1,2,5]);

  G_adj = G_adj(:,1:2,find(adc_mask),:,1:2);
  G_adj = G_adj(:,1,:,:,1) + 1i*G_adj(:,1,:,:,2);
  G_adj = reshape(G_adj,[sz*sz,sz*sz]);

end

function [SOS, phase] = TWIXtoIMG_ADJOINT(twix_obj, scanner_dict, idx)
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

sz = double(scanner_dict.sz);
sz = sz(1);
T = scanner_dict.T;
NRep = scanner_dict.NRep;
NVox = sz*sz;

adc_mask = zeros(T,1);
adc_mask(3:end-2) = 1;

% compute frequency cutoff mask for adjoint (dont use frequencies above Nyquist)
k = cumsum(squeeze(scanner_dict.grad_moms(idx,:,:,:)),1);
hsz = sz/2;

kx = k(find(adc_mask),:,1);
ky = k(find(adc_mask),:,2);

sigmask = ones(sz*sz,1);
sigmask(abs(kx(:)) > hsz) = 0;
sigmask(abs(ky(:)) > hsz) = 0;

G_adj = get_adjoint_mtx(squeeze(scanner_dict.grad_moms(idx,:,:,:)),adc_mask,T,NRep,NVox,sz);

%% Reconstruct coil images
images = zeros(size(data));
%figure;

for ii = 1:nCoils
    %images(:,:,:,ii) = fliplr(rot90(fftshift(fft2(fftshift(data(:,:,:,ii))))));
    %images(:,:,ii) = fftshift(fft2(fftshift(data(end:-1:1,:,ii))));
    
  spectrum = (data(:,:,ii));
  spectrum = spectrum(:);
%   spectrum = spectrum.*sigmask; % cut iut higher freqs
  reco = G_adj*spectrum;
  images(:,:,ii) = flipud(reshape(reco,[sz,sz]));
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



function [SOS, phase] = SIMtoIMG_NUFFT(scanner_dict,ktraj_adc,idx)
%% sort in the k-space data

data = squeeze(double(scanner_dict.all_signals(idx,3:end-2,:,1:2)));
data = data(:,:,1) + 1i*data(:,:,2);
sz=double(scanner_dict.sz(1));

%% Reconstruct coil images
images = zeros(size(data));
%figure;
ktraj_adc=ktraj_adc;
% ktraj_adc= cumsum(scanner_dict.grad_moms(idx,:,:,1:2),2);

kmax=(max(ktraj_adc(1,1:sz))-min(ktraj_adc(1,1:sz)))/2;
ktraj_adc_temp=ktraj_adc./(kmax+kmax/sz)*sz/2;

    %images(:,:,:,ii) = fliplr(rot90(fftshift(fft2(fftshift(data(:,:,:,ii))))));
    kList=double(data(end:-1:1,:));
%     kList=double(data(:,:,ii));
    [X,Y] = meshgrid(-sz/2:sz/2-1);
    ktraj_adc_temp(isnan(ktraj_adc_temp))=0;
    kList(isnan(kList))=0;
    k_regridded = griddata(ktraj_adc_temp(2,:),ktraj_adc_temp(1,:),real(kList(:)),X,Y) +1j*griddata(ktraj_adc_temp(2,:),ktraj_adc_temp(1,:),imag(kList(:)),X,Y) ;
    k_regridded(isnan(k_regridded))=0;
    images = fftshift(fft2(fftshift(k_regridded)));

% Phase images (possibly channel-by-channel and echo-by-echo)
% figure;
% imab(angle(images));colormap('jet');

sos=abs(images);
SOS=sos./max(sos(:));
phase = angle(images(:,:));

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

function [SOS, phase] = TWIXtoIMG_NUFFT(twix_obj,ktraj_adc)
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
sz=size(data,1);
%% Reconstruct coil images
images = zeros(size(data));
%figure;
ktraj_adc

kmax=(max(ktraj_adc(1,1:sz))-min(ktraj_adc(1,1:sz)))/2;
ktraj_adc_temp=ktraj_adc./(kmax+kmax/sz)*sz/2;
for ii = 1:nCoils
    %images(:,:,:,ii) = fliplr(rot90(fftshift(fft2(fftshift(data(:,:,:,ii))))));
    kList=double(data(end:-1:1,:,ii));
%     kList=double(data(:,:,ii));
    [X,Y] = meshgrid(-sz/2:sz/2-1);
    ktraj_adc_temp(isnan(ktraj_adc_temp))=0;
    kList(isnan(kList))=0;
    k_regridded = griddata(ktraj_adc_temp(2,:),ktraj_adc_temp(1,:),real(kList(:)),X,Y) +1j*griddata(ktraj_adc_temp(2,:),ktraj_adc_temp(1,:),imag(kList(:)),X,Y) ;
    k_regridded(isnan(k_regridded))=0;
    images(:,:,ii) = fftshift(fft2(fftshift(k_regridded)));
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