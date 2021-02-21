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


%% Reconstruct coil images
data_reshaped = reshape(data,size(data,1),size(data,1),50,2);
sz=size(data_reshaped)
images = zeros(sz(1),sz(2),sz(3));
%figure;
% show kspace data
data_reshaped(end,:,:,:)=0;
figure(201)
subplot(2,2,1), imagesc(abs(data_reshaped(:,:,1,1))), title('coil 1'), axis('image')
subplot(2,2,2), imagesc(abs(data_reshaped(:,:,1,2))), title('coil 2'), axis('image')

for ii = 1:nCoils
    for jj = 1:size(data_reshaped,3)
    %images(:,:,:,ii) = fliplr(rot90(fftshift(fft2(fftshift(data(:,:,:,ii))))));
    images(:,:,jj,ii) = fftshift(fft2(fftshift(data_reshaped(end:-1:1,:,jj,ii))));
    end
end

% Phase images (possibly channel-by-channel and echo-by-echo)
% figure;
% imab(angle(images));colormap('jet');

%% Sum of squares combination of channels
sos=abs(sum(images.^2,ndims(images)).^(1/2));
sos=sos./max(sos(:));
figure(101), subplot(2,1,1) 
sos2(:,:,1,:)=sos;
montage(rot90(sos2)), title('reco coilcombination'), axis('image')
subplot(2,1,2) 
plot(squeeze(sos2(end/2,end/2,:))); title('central ROI');