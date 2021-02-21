%% very simple FFT reconstruction from raw data

%% 1A. Load raw data (obtained by twix export) 
[datfilename,datpath] = uigetfile('*.dat', 'select raw data file','\\mrz3t\Upload\CEST_seq\pulseq_zero\sequences');
data_file_path=[datpath datfilename];
twix_obj = mapVBVD(data_file_path);

% sort in the k-space data
if iscell(twix_obj)
    data_unsorted = twix_obj{2}.image.unsorted();
else
    data_unsorted = twix_obj.image.unsorted();
end

% the incoming data order is [kx coils acquisitions]
data_coils_last = permute(data_unsorted, [1,3,2]);
nCoils = size(data_coils_last, 3);

data = data_coils_last;
% data(end,:,:)=0;

%% 1B. Load raw data text (using local export in pulseq_aug1)
resolution=48
[datfilename,datpath] = uigetfile('*.dat', 'select raw data file','\\mrz3t\Upload\CEST_seq\pulseq_zero\sequences');
data_file_path=[datpath datfilename];
M = importdata(data_file_path);

resolution = sqrt(size(M,1)/2)  % most probably this is the resolution (for two coil elements), but depends on your coil

Mr=M(:,1) +1j*M(:,2);
data = reshape(Mr,resolution,2,resolution);
data = permute(data, [1,3,2]);
size(data)
nCoils=2

%% 2. show kspace data
figure(101)
subplot(2,2,1), imagesc((abs(data(:,:,1)))), title('coil 1'), axis('image'); xlabel('PE/reps'); ylabel('read/T');
subplot(2,2,2), imagesc((abs(data(:,:,2)))), title('coil 2'), axis('image');  xlabel('PE'); ylabel('read');

%% 3. Reconstruct coil images

images = zeros(size(data));
%figure;

for ii = 1:nCoils
    %images(:,:,:,ii) = fliplr(rot90(fftshift(fft2(fftshift(data(:,:,:,ii))))));
    images(:,:,ii) = fftshift(fft2(fftshift(data(:,:,ii))));
    %subplot(2,2,ii);
    %imshow(abs(images(:,:,ii)), []);
    title(['RF Coil ' num2str(ii)]);
    %for ni = 1:nImages
        %tmp = abs(images(:,:,ni,ii));
        %tmp = tmp./max(tmp(:));
        %imwrite(tmp, ['img_coil_' num2str(ii) '.png'])
    %end
end

% Phase images (possibly channel-by-channel and echo-by-echo)
% figure;
% imab(angle(images));colormap('jet');

% Sum of squares combination of channels
sos=abs(sum(images.^2,ndims(images)).^(1/2));
sos=sos./max(sos(:));
figure(101), subplot(2,2,3)
imagesc(sos), title('reco coilcombination, PE='), axis('image'); xlabel('PE'); ylabel('read');
subplot(2,2,4)
angle_img = angle(images(:,:,2));
imagesc(angle_img); title('reco coilcombination PHASE, PE='), axis('image'); xlabel('PE'); ylabel('read');

