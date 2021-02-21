%% very simple FFT reconstruction from raw data
addpath('C:\Users\danghi\Documents\Matlab\cest_eval\load_fcns\reco_raw\mapVBVD')
%% 1A. Load raw data (obtained by twix export) 
[datfilename,datpath] = uigetfile('*.dat', 'select raw data for no prep measurement','\\141.67.249.47\MRTransfer\pulseq_zero\sequences');
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

data0 = data_coils_last;
% data(end,:,:)=0;

[datfilename,datpath] = uigetfile('*.dat', 'select raw data for 1DEG Prep measurement','\\141.67.249.47\MRTransfer\pulseq_zero\sequences');
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

data1 = data_coils_last;
% data(end,:,:)=0;

%% 1B. Load raw data text (using local export in pulseq_aug1)
% resolution=32;
% nCoils=20;
% 
% [datfilename,datpath] = uigetfile('*.dat', 'select raw data file','\\141.67.249.47\MRTransfer\pulseq_zero\sequences');
% data_file_path=[datpath datfilename];
% M = importdata(data_file_path);
% 
% Mr=M(:,1) +1j*M(:,2);
% data00 = reshape(Mr,resolution+4,nCoils,resolution);
% data00 = data00(1:resolution,:,:);
% data00 = permute(data00, [1,3,2]);
% 
% [datfilename,datpath] = uigetfile('*.dat', 'select raw data file','\\141.67.249.47\MRTransfer\pulseq_zero\sequences');
% data_file_path=[datpath datfilename];
% M = importdata(data_file_path);
% 
% Mr=M(:,1) +1j*M(:,2);
% data11 = reshape(Mr,resolution+4,nCoils,resolution);
% data11 = data11(1:resolution,:,:);
% data11 = permute(data11, [1,3,2]);

%% 2. show kspace data
figure(101)
title('Phase in kspace, No Prep Pulse')
for i=1:nCoils
    subplot(4,5,i), imagesc((angle(data0(:,:,i)))), title(['coil ', num2str(i)]), axis('image'); xlabel('PE'); ylabel('read');
end

figure(102)
title('Phase in kspace, 1Deg Prep Pulse')
for i=1:nCoils
    subplot(4,5,i), imagesc((angle(data1(:,:,i)))), title(['coil ', num2str(i)]), axis('image'); xlabel('PE'); ylabel('read');
end

figure(103)
for i=1:nCoils
    subplot(4,5,i), imagesc((abs(data0(:,:,i)./data1(:,:,i)))), title(['coil ', num2str(i)]), axis('image'); xlabel('PE'); ylabel('read');
end
%% 3. Reconstruct coil images

images0 = zeros(size(data0));
%figure;

permvec = zeros(resolution,1);
permvec(1) = 0;
for i=1:resolution/2
    permvec(i*2) = (-i);
    if i < resolution/2
        permvec(i*2+1) = i;
    end
end
permvec = permvec + resolution/2 + 1;   

inverse_perm = (1:32);
[~,sort_idx] = sort(permvec);
inverse_perm = inverse_perm(sort_idx);

data0 = data0(:,inverse_perm,:);
for ii = 1:nCoils
    images0(:,:,ii) = fftshift(fft2(fftshift(data0(:,:,ii))));
    subplot(5,4,ii);
    imshow(abs(images0(:,:,ii)), []);
    title(['RF Coil ' num2str(ii)]);
end

data1 = data1(:,inverse_perm,:);
images1 = zeros(size(data1));
for ii = 1:nCoils
    images1(:,:,ii) = fftshift(fft2(fftshift(data1(:,:,ii))));
    subplot(5,4,ii);
    imshow(abs(images1(:,:,ii)), []);
    title(['RF Coil ' num2str(ii)]);
end

% Sum of squares combination of channels
sos0=abs(sum(images0.^2,ndims(images0)).^(1/2));
sos0=sos0./max(sos0(:));
figure(104), subplot(2,2,3)
imagesc(sos0), title('reco SOS coilcombination, No Prep Pulse'), axis('image'); xlabel('PE'); ylabel('read'); colorbar();
subplot(2,2,4)
angle_img0 = angle(images0(:,:,2));
imagesc(angle_img0); title('reco coilcombination PHASE, No Prep Pulse'), axis('image'); xlabel('PE'); ylabel('read');

sos1=abs(sum(images1.^2,ndims(images1)).^(1/2));
sos1=sos1./max(sos1(:));
figure(105), subplot(2,2,3)
imagesc(sos1), title('reco SOS coilcombination, 1DEG Prep Pulse'), axis('image'); xlabel('PE'); ylabel('read'); colorbar();
subplot(2,2,4)
angle_img1 = angle(images1(:,:,2));
imagesc(angle_img1); title('reco SOS coilcombination PHASE, 1DEG Prep Pulse'), axis('image'); xlabel('PE'); ylabel('read');

