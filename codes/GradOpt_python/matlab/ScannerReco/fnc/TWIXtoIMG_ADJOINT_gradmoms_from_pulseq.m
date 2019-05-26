function [SOS, phase] = TWIXtoIMG_ADJOINT_gradmoms_from_pulseq(twix_obj, ktraj_adc)
%% sort in the k-space data
  %% sort in the k-space data
  if iscell(twix_obj)
      raw_kspace = twix_obj{2}.image();
  else
      raw_kspace = twix_obj.image();
  end
  
  % the incoming data order is [kx coils acquisitions]
  raw_kspace = permute(raw_kspace, [1,3,2]);
  nCoils = size(raw_kspace, 3);

  sz=size(raw_kspace,1);

  % compute frequency cutoff mask for adjoint (dont use frequencies above Nyquist)

deltak=1000/twix_obj{1, 2}.hdr.Meas.ReadFoV;
ktraj_adc=ktraj_adc/deltak;

%  DENKMAL! HOLOCAUST
%   kmax=(max(ktraj_adc(1,1:sz))-min(ktraj_adc(1,1:sz)))/2;  
%   ktraj_adc=ktraj_adc./(kmax+kmax/sz)*sz/2;  
  
  ktraj_adc = reshape(ktraj_adc(1:2,:),[2,sz,sz]);
  k = permute(ktraj_adc,[2,3,1]);
  
  hsz = sz/2;
  kx = k(:,:,1);
  ky = k(:,:,2);
  sigmask = ones(sz*sz,1);
  sigmask(abs(kx(:)) > hsz) = 0;
  sigmask(abs(ky(:)) > hsz) = 0;  
  
  %keyboard

  G_adj = get_adjoint_mtx(squeeze(k));

  %% Reconstruct coil images
  images = zeros(size(raw_kspace));
% raw_kspace=circshift(raw_kspace,[sz/2+4,0,0]);
  for ii = 1:nCoils
    spectrum = raw_kspace(:,:,ii);
    reco = G_adj*(spectrum(:));
    images(:,:,ii) = (reshape(reco,[sz,sz]));
  end

  images =  permute(images(end:-1:1,end:-1:1,:),[2,1,3]);
      
  sos=abs(sum(images.^2,ndims(images)).^(1/2));
  SOS=sos./max(sos(:));
  phase = angle(images(:,:,ii));

end

function [G_adj] = get_adjoint_mtx(k)
 
  sz = size(k);
  sz = sz(1);
  NVox = sz*sz;

  G_adj = zeros(sz,sz,NVox,3,3, 'single');
  G_adj(:,:,:,3,3) = 1;

  % get ramps
  baserampX = linspace(-1,1,sz + 1);
  baserampY = linspace(-1,1,sz + 1);

  rampX = pi*baserampX;
  rampX = rampX(1:sz).'*ones(1,sz);
  rampX = reshape(rampX,[1,1,NVox]);

  rampY = pi*baserampY;
  rampY = ones(sz,1)*rampY(1:sz);
  rampY = reshape(rampY,[1,1,NVox]);

  B0X = reshape(k(:,:,1), [sz,sz,1]) .* rampY;
  B0Y = reshape(k(:,:,2), [sz,sz,1]) .* rampX;

  B0_grad = reshape((B0X + B0Y), [sz,sz,NVox]);

  B0_grad_adj_cos = cos(B0_grad);
  B0_grad_adj_sin = sin(B0_grad);      

  % adjoint
  G_adj(:,:,:,1,1) = single(B0_grad_adj_cos);
  G_adj(:,:,:,1,2) = single(B0_grad_adj_sin);
  G_adj(:,:,:,2,1) = -single(B0_grad_adj_sin);
  G_adj(:,:,:,2,2) = single(B0_grad_adj_cos);
 
  G_adj = single(G_adj);
  G_adj = permute(G_adj,[3,4,1,2,5]);

  G_adj = G_adj(:,1:2,:,:,1:2);
  G_adj = G_adj(:,1,:,:,1) + 1i*G_adj(:,1,:,:,2);
  G_adj = reshape(G_adj,[sz*sz,sz*sz]);

end