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
imagesc((sos)), title('reco coilcombination'), axis('image')

show kspace data
figure(101)
subplot(2,2,1), imagesc((abs(data(:,:,1)))), title('coil 1'), axis('image')
subplot(2,2,2), imagesc((abs(data(:,:,2)))), title('coil 2'), axis('image')
end

end