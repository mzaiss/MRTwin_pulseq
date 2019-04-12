%% very simple FFT reconstruction from raw data

%% Load raw data
[datfilename,datpath] = uigetfile('*.dat', 'select raw data file','\\mrz3t\Upload\CEST_seq\pulseq_zero\sequences');
data_file_path=[datpath datfilename];

twix_obj = mapVBVD(data_file_path);


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

%% show kspace data
figure(101)
subplot(2,2,1), imagesc(rot90(abs(data(:,:,1)))), title('coil 1'), axis('image')
subplot(2,2,2), imagesc(rot90(abs(data(:,:,2)))), title('coil 2'), axis('image')

%% Reconstruct coil images

images = zeros(size(data));
%figure;

for ii = 1:nCoils
    %images(:,:,:,ii) = fliplr(rot90(fftshift(fft2(fftshift(data(:,:,:,ii))))));
    images(:,:,ii) = fftshift(fft2(fftshift(data(end:-1:1,:,ii))));
    %subplot(2,2,ii);
    %imshow(abs(images(:,:,ii)), []);
    %title(['RF Coil ' num2str(ii)]);
    %for ni = 1:nImages
        %tmp = abs(images(:,:,ni,ii));
        %tmp = tmp./max(tmp(:));
        %imwrite(tmp, ['img_coil_' num2str(ii) '.png'])
    %end
end

% Phase images (possibly channel-by-channel and echo-by-echo)
% figure;
% imab(angle(images));colormap('jet');

%% Sum of squares combination of channels
sos=abs(sum(images.^2,ndims(images)).^(1/2));
sos=sos./max(sos(:));
figure(101), subplot(2,2,3)
imagesc(rot90(sos)), title('reco coilcombination'), axis('image')