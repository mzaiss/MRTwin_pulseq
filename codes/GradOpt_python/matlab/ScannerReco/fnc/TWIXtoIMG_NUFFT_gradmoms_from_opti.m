% regrid measured data from twix object from nonuniform to Cartesian grid
% to compute k-space locations for gridding, use gradient moments generated during optimization

% call with
% [sos, phase] = TWIXtoIMG_NUFFT_gradmoms_from_opti(twix_obj,scanner_dict.grad_moms(ii,:,:,:));

function [SOS, phase] = TWIXtoIMG_NUFFT_gradmoms_from_opti(twix_obj,grad_moms)

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
  NRep=size(grad_moms,2);
  grad_moms = squeeze(grad_moms(:,:,1:2));
  grad_moms = cat(1,zeros(1,NRep,2),grad_moms);
  ktraj_adc_sim = squeeze(cumsum(grad_moms(1:end-1,:,:),1));
  ktraj_adc_sim = ktraj_adc_sim(3:end-2,:,:);
  ktraj_adc_temp = double(reshape(permute(ktraj_adc_sim,[3,2,1]),2,[]));
  
  for ii = 1:nCoils
    % transpose
    kList=double(raw_kspace(:,:,ii)).';
    [X,Y] = meshgrid(-sz/2:sz/2-1);
    
    k_regridded = griddata(ktraj_adc_temp(1,:),ktraj_adc_temp(2,:),real(kList(:)),X,Y) +1j*griddata(ktraj_adc_temp(1,:),ktraj_adc_temp(2,:),imag(kList(:)),X,Y) ;
    k_regridded(isnan(k_regridded))=0;
    images(:,:,ii) = fftshift(fft2(fftshift(k_regridded.')));
  end
  
  sos=abs(sum(images.^2,ndims(images)).^(1/2));
  SOS=sos./max(sos(:));
  phase = angle(images(:,:,ii));

end