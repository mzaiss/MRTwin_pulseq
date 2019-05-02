function [SOS, phase] = TWIXtoIMG_NUFFT_gradmoms_from_pulseq(twix_obj,ktraj_adc)


%   keyboard

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

%   kmax=(max(ktraj_adc(1,1:sz))-min(ktraj_adc(1,1:sz)))/2;
%   ktraj_adc_temp=ktraj_adc./(kmax+kmax/sz)*sz/2;
  
  deltak=1000/twix_obj{1, 2}.hdr.Meas.ReadFoV;
  ktraj_adc=ktraj_adc/deltak;

  % permute and reshape to match k-space orientation
   ktraj_adc_temp = ktraj_adc; % reshape(permute(reshape(ktraj_adc(1:2,:),[2,sz,sz]),[1,3,2]),[],sz*sz);
try
  for ii = 1:nCoils
    % transpose
      kList=(double(raw_kspace(:,:,ii)));
  %     kList=double(data(:,:,ii));
      [X,Y] = meshgrid(-sz/2:sz/2-1);
      ktraj_adc_temp(isnan(ktraj_adc_temp))=0;
      kList(isnan(kList))=0;
      %k_regridded = griddata(ktraj_adc_temp(2,:),ktraj_adc_temp(1,:),real(kList(:)),X,Y) +1j*griddata(ktraj_adc_temp(2,:),ktraj_adc_temp(1,:),imag(kList(:)),X,Y) ;
      k_regridded = griddata(ktraj_adc_temp(1,:),ktraj_adc_temp(2,:),real(kList(:)),X,Y) +1j*griddata(ktraj_adc_temp(1,:),ktraj_adc_temp(2,:),imag(kList(:)),X,Y) ;
      k_regridded(isnan(k_regridded))=0;
      %images(:,:,ii) = fftshift(fft2(fftshift(k_regridded)));
      images(:,:,ii) = fftshift(fft2(fftshift(k_regridded)));
  end
  
%     images =  permute(images(end:-1:1,end:-1:1,:),[2,1,3]);
    images =  permute(images(:,:,:),[2,1,3]);
    
catch
    warning('gridata probably died because of singleton support points');
end
  % Phase images (possibly channel-by-channel and echo-by-echo)
  % figure;
  % imab(angle(images));colormap('jet');

  sos=abs(sum(images.^2,ndims(images)).^(1/2));
  SOS=sos./max(sos(:));
  phase = angle(images(:,:,ii));

end