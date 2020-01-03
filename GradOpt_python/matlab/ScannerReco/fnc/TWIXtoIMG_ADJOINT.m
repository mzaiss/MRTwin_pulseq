function [SOS, phase] = TWIXtoIMG_ADJOINT(twix_obj, scanner_dict, idx)

if iscell(twix_obj)
    raw_kspace = twix_obj{2}.image();
else
    raw_kspace = twix_obj.image();
end

% the incoming data order is [kx coils acquisitions]
raw_kspace = permute(raw_kspace, [1,3,2]);
nCoils = size(raw_kspace, 3);

data = raw_kspace;

sz = double(scanner_dict.sz);
sz = sz(1);
T = scanner_dict.T;
NRep = scanner_dict.NRep;
NVox = sz*sz;

adc_mask = zeros(T,1);
adc_mask(3:end-2) = 1;

% compute frequency cutoff mask for adjoint (dont use frequencies above Nyquist)
grad_moms = scanner_dict.grad_moms(idx,:,:,:);
grad_moms = cat(1,zeros(1,NRep,2),grad_moms);
k = cumsum(grad_moms(1:end-1,:,:),1);
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

for ii = 1:nCoils-1
    %images(:,:,:,ii) = fliplr(rot90(fftshift(fft2(fftshift(data(:,:,:,ii))))));
    %images(:,:,ii) = fftshift(fft2(fftshift(data(end:-1:1,:,ii))));
    
    spectrum = (data(:,:,ii));
    spectrum = spectrum(:);
    
    reco = G_adj*spectrum;
    images(:,:,ii) = flipud(reshape(reco,[sz,sz]));
end
images =  permute(images(end:-1:1,end:-1:1,:),[2,1,3]);

% Phase images (possibly channel-by-channel and echo-by-echo)
% figure;
% imab(angle(images));colormap('jet');

sos=abs(sum(images.^2,ndims(images)).^(1/2));
SOS=sos./max(sos(:));
phase = angle(images(:,:,ii));

if 0
    %% Sum of squares combination of channels
    figure(101), subplot(2,2,3)
    imagesc(sos), title('reco coilcombination'), axis('image')
    
    show kspace data
    figure(101)
    subplot(2,2,1), imagesc((abs(data(:,:,1)))), title('coil 1'), axis('image')
    subplot(2,2,2), imagesc((abs(data(:,:,2)))), title('coil 2'), axis('image')
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
rampX = rampX(1:sz).'*ones(1,sz);
rampX = reshape(rampX,[1,1,NVox]);

rampY = pi*baserampY;
rampY = ones(sz,1)*rampY(1:sz);
rampY = reshape(rampY,[1,1,NVox]);

B0X = reshape(k(:,:,1), [T,NRep,1]) .* rampY;
B0Y = reshape(k(:,:,2), [T,NRep,1]) .* rampX;

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