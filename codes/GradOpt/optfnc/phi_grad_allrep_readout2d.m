% Minimize loss: ||m-E.T*Em||_2 + lambda*||cumsum(gx/y)||_2
% and return objective value phi, and derivative dg
%
% E - E = exp(1i*sum(gx*rampY + gy*rampY))
%
% m - magnetization/image in the spatial domain
% gx/gy - temporal gradient waveforms
% rampX - fixed gradient spatial form in x dir
% rampY ...
% sz - image size
% lambda - regularization weight controlling the grad_moms penalty
% use_tanh_grad_moms_cap - flag that uses hard tanh constrain on the grad_moms: grad_moms = tanh(cumsum(gx/y))
% both L2 and tanh help prevent the case where the kspace locations reached are above the frequencies of Nyquist condition

% returns
% phi - achieved loss value
% dg - analytical derivative of the gradient variable
% prediction - reconstructed image
% E - encoding matrix
% grad_moms - integral (cumsum) over the gradients
% effective gradients (computed as finite derivative of the grad_moms)

function [phi,tdg,prediction,E,grad_moms,grads] = phi_grad_allrep_readout2d(g,m,rampX,rampY,adc_mask,sz,NRep,lambda,use_tanh_grad_moms_cap)

g = reshape(g,NRep,[],2);

% set the number of time points in the readout
T = size(g,2);

nfact = (prod(sz));                                                                            % Fourier transform normalization coefficient                                                                     

% L2 norm for regularization
l2  = @(z) z.*conj(z);
dl2 = @(z) 2*z;
dtanh = @(z) 1 - tanh(z).^2;

% vectorize ramps and the image
rampX = rampX(:).'; rampY = rampY(:).'; m = m(:);

phi = 0;
tdg = zeros(NRep,2*T);

prediction = 0;

grads = cell(NRep,1);
for rep = 1:NRep

  % integrate over time to get grad_moms from the gradients
  grad_moms = cumsum(squeeze(g(rep,:,:)),1);

  % prewinder (to be relaxed in future)
  %g(:,1) = g(:,1) - T/2 - 1;

  if use_tanh_grad_moms_cap
    fmax = sz / 2;                                                                                       % cap the grad_moms to [-1..1]*sz/2

    for i = 1:2
      grad_moms(:,i) = fmax(i)*tanh(grad_moms(:,i));                                                                        % soft threshold
      grad_moms(abs(grad_moms(:,i)) > fmax(i),i) = sign(grad_moms(abs(grad_moms(:,i)) > fmax(i),i))*fmax(i);  % hard threshold, this part is nondifferentiable
    end
  end

  grads{rep} = diff(cat(1,[0,0],grad_moms),1);                      % actual gradient forms are the derivative of grad_moms (inverse cumsum)

  % compute the B0 by adding gradients in X/Y after multiplying them respective ramps
  B0X = grad_moms(:,1) * rampX; B0Y = grad_moms(:,2) * rampY;

  B0 = B0X + B0Y;

  % encoding operator (ignore relaxation)
  E = exp(1i*B0);
  
  E = E .* adc_mask;
  
  % compute loss
  prediction = prediction + (E'*E)*m / nfact;
  
  phi = phi + lambda*sum(l2(grad_moms(:)));
  
end

loss = (prediction - m);
phi = phi + sum(l2(loss));

cmx = conj(dl2(loss)) * m.' / nfact ;


for rep = 1:NRep
  
  % integrate over time to get grad_moms from the gradients
  grad_moms = cumsum(squeeze(g(rep,:,:)),1);

  % prewinder (to be relaxed in future)
  %g(:,1) = g(:,1) - T/2 - 1;

  save_grad_moms = grad_moms;                                                                                   % needed for tanh derivative

  if use_tanh_grad_moms_cap
    fmax = sz / 2;                                                                                       % cap the grad_moms to [-1..1]*sz/2

    for i = 1:2
      grad_moms(:,i) = fmax(i)*tanh(grad_moms(:,i));                                                                                  % soft threshold
      grad_moms(abs(grad_moms(:,i)) > fmax(i),i) = sign(grad_moms(abs(grad_moms(:,i)) > fmax(i),i))*fmax(i);  % hard threshold, this part is nondifferentiable
    end
  end

  % compute the B0 by adding gradients in X/Y after multiplying them respective ramps
  B0X = grad_moms(:,1) * rampX; B0Y = grad_moms(:,2) * rampY;

  B0 = B0X + B0Y;

  % encoding operator (ignore relaxation)
  E = exp(1i*B0);
  
  E = E .* adc_mask;

  % compute derivative with respect to temporal gradient waveforms
  dgXY = (conj(E) * cmx + conj(E * cmx.')) .* E;

  dg = zeros(size(grad_moms));
  dg(:,1) = sum(1i*dgXY.*rampX,2);
  dg(:,2) = sum(1i*dgXY.*rampY,2);

  if use_tanh_grad_moms_cap
    for i = 1:2
      dg(:,i) = fmax(i)*dtanh(save_grad_moms(:,i)) .* dg(:,i);
    end
  end

  dg = cumsum(dg, 1, 'reverse');
  dg = real(dg(:));

  % regularization part derivatives
  rega = dl2(grad_moms);
  if use_tanh_grad_mom_scap
    for i = 1:2
      rega(:,i) = fmax(i)*dtanh(save_grad_moms(:,i)) .* rega(:,i);
    end  
  end

  rega = lambda*cumsum(rega, 1, 'reverse');

  tdg(rep,:) = dg + rega(:);

end

tdg = tdg(:);

