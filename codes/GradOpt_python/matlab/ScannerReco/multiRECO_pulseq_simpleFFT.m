function multiRECO_pulseq_simpleFFT
%% very simple FFT reconstruction from raw data
addpath('D:\root\ZAISS_LABLOG\LOG_MPI\0_CESTtool\CEST_EVAL_GLINT\reco_raw\recoVBVD')
addpath('D:\root\ZAISS_LABLOG\LOG_MPI\0_CESTtool\CEST_EVAL_GLINT\reco_raw\mapVBVD')
%% Load raw data
[filename,rawpname] = uigetfile({'*.dat'},['select raw fils for reconstruction'],'MultiSelect','on','\\mrz3t\Upload\CEST_seq\MRIZERO_data');
if iscell(filename)==0
    temp=filename; clear filename;
    filename{1}=temp; clear temp;
end

origpath=pwd;

set(0, 'DefaultLineLineWidth', 2);
figure(1)
array=1:numel(filename);

twix_obj = mapVBVD([rawpname '/' filename{1}]);
[sos_base, phase_base] = TWIXtoIMG(twix_obj);

for ii=array
    
twix_obj = mapVBVD([rawpname '/' filename{ii}]);
[sos, phase] = TWIXtoIMG(twix_obj);

subplot(2,2,1), imagesc(rot90(sos),[0 1]), title(sprintf('reco sos, iter %d',ii)), axis('image'); colorbar;
subplot(2,2,3), imagesc(rot90(phase)), title('reco phase coil(1) '), axis('image'); colorbar;
subplot(2,2,2), imagesc(rot90(sos_base),[0 1]), title(sprintf('MEAS reco sos, iter %d',1)), axis('image'); colorbar;
subplot(2,2,4), imagesc(rot90(phase_base)), title('reco phase coil(1) '), axis('image'); colorbar;
set(gcf, 'Outerposition',[404   356   850   592])

% create gif (out.gif)
drawnow
      frame = getframe(1);
      im = frame2im(frame);
      im_MEAS(:,:,:,ii)=im;
      [imind,cm] = rgb2ind(im,32);
      if ii == 1
          imwrite(imind,cm,'out.gif','gif', 'Loopcount',inf);
      elseif ii==numel(array)
          imwrite(imind,cm,'out.gif','gif','WriteMode','append','DelayTime',3);
      else
          imwrite(imind,cm,'out.gif','gif','WriteMode','append','DelayTime',0.0005);
      end
end
set(0, 'DefaultLineLineWidth', 0.5);
last_fn_meas=filename{ii};
save('togif.mat','im_MEAS','last_fn_meas');
end


function [SOS, phase] = TWIXtoIMG(twix_obj)
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