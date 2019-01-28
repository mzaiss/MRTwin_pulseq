
clear all; close all;

addpath  ../SequenceSIM/3rdParty/pulseq-master/matlab/
addpath optfnc

% forward Fourier transform
fftfull =  @(x) ifftshift(fftn(fftshift(x)))/sqrt(numel(x));
ifftfull =  @(x) ifftshift(ifftn(fftshift(x)))*sqrt(numel(x));

% NRMSE error function
e = @(utrue,u) 100*norm(u(:)-utrue(:))/norm(utrue(:));

%% Check numerical vs. analytical derivatives : optimize all reps simultaneously

use_gpu = 0;

T = 4;       % number of time points in readout                                                                                                  
sz = [4, 6]; % image size (Nx Ny)          
Nrep = 8;

% set gradient spatial forms
rampX = pi*linspace(-1,1,sz(1) + 1);
rampX = rampX(1:end-1);
rampX = repmat(rampX.', [1, sz(2)]);

% set gradient spatial forms
rampY = pi*linspace(-1,1,sz(2) + 1);
rampY = rampY(1:end-1);
rampY = repmat(rampY, [sz(1), 1]);

adc_mask = ones(T,1); adc_mask(1:2) = 0;

% initialize gradients (X/Y directions)
g = rand(Nrep,T,2); g = g(:);

% initialize complex-valued magnetization image
m = rand(sz(1),sz(2)) + 1i*rand(sz(1),sz(2));

use_tanh_grad_moms_cap = 1;                                                 % otherwise put L2 penalty on the grad_moms controlled by lambda
lambda = 0*1e-2;

% pack the parameters for the gradient function
args = {m,rampX,rampY,adc_mask,sz,Nrep,lambda,use_tanh_grad_moms_cap,use_gpu};
[phi,dg_ana] = phi_grad_allrep_readout2d(g(:),args{:}); % compute loss and analytical derivatives

% compute numerical derivatives
h = 1e-4;
dg = zeros(size(g)); dphi_h = zeros(size(g));
for i=1:numel(g)
  dg(i) = 1; dphi_h(i) = phi_grad_allrep_readout2d(g+h*dg,args{:})-phi;
  dg(i) = 0;
  
  if mod(i,100) == 0
    dx_num = dphi_h/h;
    fprintf('deriv-err=%1.3f%%\n',e(dx_num(1:i),dg_ana(1:i)))
  end
end
dx_num = dphi_h/h; clear dphi_h i dx

fprintf('deriv-err=%1.3f%%\n',e(dx_num,dg_ana(:)))

% compare analytical and numerical gradients
%[dg_ana(:), dx_num]

%% optimize all reps simultaneously
%close all;

use_gpu = 1;

% params
NRep = 16;                                                                                                           % number of repetitions
sz = [16,16];                                                                                                                   % image size
T = sz(2);                                                                                        % set the number of time points in readout
nmb_rand_restarts = 1;                                                                      % number of restarts with random initializations
do_versbose = 0;                                                                         % additionally show learned Fourier basis functions

% regularization parameters
use_tanh_grad_moms_cap = 1;                                                            % limit the effective grad_moms to sz*[-1..1]/2 range
lambda = 0*1e-6;                                                                                           % put L2 penalty on the grad_moms

gtruth_m = load('../../data/phantom.mat'); gtruth_m = gtruth_m.phantom;
gtruth_m = imresize(gtruth_m,sz);  % resize to something managable
 
%gtruth_m = fftfull(gtruth_m); gtruth_m(8:end,:) = 0; gtruth_m = ifftfull(gtruth_m);        % set some part of kspace to zero just for a test
                                                                     
% set the optimizer
p = struct();
nMVM = 100;  % number of optimization iterations
p.length = -nMVM;
p.method = 'LBFGS';

% set ADC mask
adc_mask = ones(T,1); adc_mask(1:4) = 1;

% set gradient spatial forms
rampX = pi*linspace(-1,1,sz(1) + 1);
rampX = rampX(1:end-1);
rampX = repmat(rampX.', [1, sz(2)]);

rampY = pi*linspace(-1,1,sz(2) + 1);
rampY = rampY(1:end-1);
rampY = repmat(rampY, [sz(1), 1]);

minerr = 1e8;

% initialize reconstructed image
all_grad = cell(NRep,1);                                                                         % learned gradients at all repetition steps

for rnd_restart = 1:nmb_rand_restarts

  % initialize gradients
  %g = zeros(T,2); g = g(:);                                                                       % initialize the gradients to all zeros
  %g = zeros(NRep,T,2); g(:,:,1) = rand(NRep,T,1) - 0.5; g = g(:);
  g = 1.0*(rand(NRep,T,2) - 0.5); g = g(:); 
  
  p.length = -100;

  % do optimization for g of E(g), loss --> (||error_m - E.T*E*error_m||^2 + lambda*||cumsum(g)||^2) 
  if use_gpu
    args = {gpuArray(single(gtruth_m)),gpuArray(single(rampX)),gpuArray(single(rampY)),gpuArray(single(adc_mask)),sz,NRep,lambda,use_tanh_grad_moms_cap,use_gpu};
  else
    args = {gtruth_m,rampX,rampY,adc_mask,sz,NRep,lambda,use_tanh_grad_moms_cap,use_gpu};
  end  
  
  [g,phi] = minimize(g(:),'phi_grad_allrep_readout2d',p,args{:});

  % select the gradients with the lowest loss achieved
  phi = phi(end);
  if phi < minerr
    bestgrad = g;
    minerr = phi;
  end
end

p.length = -nMVM;


%%
close all

grad_moms = zeros(NRep,T,2);
grad_moms(:,:,1) = repmat(linspace(-sz(1)/2,sz(1)/2,sz(1)),[sz(1),1,1]);
grad_moms(:,:,2) = repmat(linspace(-sz(2)/2,sz(2)/2,sz(2)).',[1 sz(2),1]);

%grad_moms = grad_moms + 0.2*randn(size(grad_moms));

bestgrad = diff(cat(2,zeros(sz(1),1,2),grad_moms),[],2); use_tanh_grad_moms_cap = 0;

g = bestgrad;

lambda = 0;

args = {gtruth_m,rampX,rampY,adc_mask,sz,NRep,lambda,use_tanh_grad_moms_cap,0};
[~,~,reco_current,E,grad_moms_output,grads] = phi_grad_allrep_readout2d(g(:),args{:});

figure(3), hold off; plot(1);

for rep = 1:NRep
  
  figure(1), plot(grads{rep}); title(['learned gradients at repetition ', num2str(rep), ' blue - grad X, orange - grad Y']); xlabel('time'); ylabel('gradient strength (au)');
  
  figure(2),
    subplot(1,2,1), imagesc(abs(reshape(reco_current,sz))); title(['repetition ',num2str(rep),' :prediction ',' : reconstruction, error=',num2str(e(gtruth_m(:),reco_current(:)))]);
    subplot(1,2,2), imagesc(abs(gtruth_m)); title('global target to predict (actual image)');
    
  grad_moms_plot = cumsum(grads{rep},1);
  %grad_moms_plot = squeeze(grad_moms(rep,:,:));
    
  % plot actual sampled kspace locations  
  figure(3)
  c = ones(T,1)*rep;                                                                                                % color code repetitions
    hold on; scatter(grad_moms_plot(:,2), grad_moms_plot(:,1),[],c); hold off; axis([-sz(1)/2,sz(1)/2,-sz(2)/2,sz(2)/2]); title('kspace sampled locations'); xlabel('readout direction'); ylabel('phase encode direction');
      
  %pause
end

learned_grads = grads;





























