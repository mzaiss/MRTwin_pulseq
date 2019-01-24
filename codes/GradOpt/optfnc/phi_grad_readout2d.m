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
% lambda - regularization weight controlling the field penalty
% use_tanh_fieldcap - flag that uses hard tanh constrain on the field: field = tanh(cumsum(gx/y))
% both L2 and tanh help prevent the case where the kspace locations reached are above the frequencies of Nyquist condition

% returns
% phi - achieved loss value
% dg - analytical derivative of the gradient variable
% prediction - reconstructed image
% E - encoding matrix
% field - integral (cumsum) over the gradients
% effective gradients (computed as finite derivative of the field)

function [phi,dg,prediction,E,field,grads] = phi_grad_readout2d(g,m,rampX,rampY,sz,lambda,use_tanh_fieldcap)

g = reshape(g,[],2);

% set the number of time points in the readout
T = size(g,1);

nfact = (prod(sz));                                                                            % Fourier transform normalization coefficient                                                                     

% L2 norm for regularization
l2  = @(z) z.*conj(z);
dl2 = @(z) 2*z;
dtanh = @(z) 1 - tanh(z).^2;

% vectorize ramps and the image
rampX = rampX(:).'; rampY = rampY(:).'; m = m(:);

% integrate over time to get field from the gradients
field = cumsum(g,1);

% prewinder (to be relaxed in future)
%g(:,1) = g(:,1) - T/2 - 1;

save_field = field;                                                                                             % needed for tanh derivative

if use_tanh_fieldcap
  fmax = sz / 2;                                                                                             % cap the field to [-1..1]*sz/2
  
  for i = 1:2
    field(:,i) = fmax(i)*tanh(field(:,i));                                                                                  % soft threshold
    field(abs(field(:,i)) > fmax(i),i) = sign(field(abs(field(:,i)) > fmax(i),i))*fmax(i);  % hard threshold, this part is nondifferentiable
  end
end

grads = diff(cat(1,[0,0],field),1);                                     % actual gradient forms are the derivative of field (inverse cumsum)

% compute the B0 by adding gradients in X/Y after multiplying them respective ramps
B0X = field(:,1) * rampX; B0Y = field(:,2) * rampY;

B0 = B0X + B0Y;

% encoding operator (ignore relaxation)
E = exp(1i*B0);

% compute loss
prediction = (E'*E)*m / nfact;
loss = (prediction - m);
phi = sum(l2(loss));

phi = phi + lambda*sum(l2(field(:)));

% compute derivative with respect to temporal gradient waveforms
cmx = conj(dl2(loss)) * m.' / nfact ;
dgXY = (conj(E) * cmx + conj(E * cmx.')) .* E;

dg = zeros(size(field));
dg(:,1) = sum(1i*dgXY.*rampX,2);
dg(:,2) = sum(1i*dgXY.*rampY,2);

if use_tanh_fieldcap
  for i = 1:2
    dg(:,i) = fmax(i)*dtanh(save_field(:,i)) .* dg(:,i);
  end
end

dg = cumsum(dg, 1, 'reverse');
dg = real(dg(:));

% regularization part derivatives
rega = dl2(field);
if use_tanh_fieldcap
  for i = 1:2
    rega(:,i) = fmax(i)*dtanh(save_field(:,i)) .* rega(:,i);
  end  
end

rega = lambda*cumsum(rega, 1, 'reverse');

dg = dg + rega(:);



