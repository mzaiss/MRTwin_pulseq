% regrid measured data from twix object from nonuniform to Cartesian grid
% to compute k-space locations for gridding, use effective gradient moments output by pulseq

function [SOS, phase] = TWIXtoIMG_NUFFT_gradmoms_from_pulseq(twix_obj,ktraj_adc)

  %% sort in the k-space data
  if iscell(twix_obj)
      raw_kspace = twix_obj{2}.image();
  else
      raw_kspace = twix_obj.image();
  end
  
  % the incoming data order is [kx coils acquisitions]
  raw_kspace = permute(raw_kspace, [1, 3, 2]);
  nCoils = size(raw_kspace, 3);

  sz=size(raw_kspace,1);
  
  %% Reconstruct coil images
  images = zeros(size(raw_kspace));

  kmax=(max(ktraj_adc(1,1:sz))-min(ktraj_adc(1,1:sz)))/2;
  ktraj_adc_temp=ktraj_adc./(kmax+kmax/sz)*sz/2;

  % permute and reshape to match k-space orientation
  ktraj_adc_temp = reshape(permute(reshape(ktraj_adc_temp(1:2,:),[2,sz,sz]),[1,3,2]),[],sz*sz);

  for ii = 1:nCoils
    % transpose
    kList=double(raw_kspace(:,:,ii)).';
    [X,Y] = meshgrid(-sz/2:sz/2-1);
    
    k_regridded = griddata(ktraj_adc_temp(1,:),ktraj_adc_temp(2,:),real(kList(:)),X,Y) +1j*griddata(ktraj_adc_temp(1,:),ktraj_adc_temp(2,:),imag(kList(:)),X,Y) ;
    k_regridded(isnan(k_regridded))=0;
    images(:,:,ii) = fftshift(fft2(fftshift(fliplr(flipud(k_regridded.')))));
  end
  
  sos=abs(sum(images.^2,ndims(images)).^(1/2));
  SOS=sos./max(sos(:));
  phase = angle(images(:,:,ii));

end