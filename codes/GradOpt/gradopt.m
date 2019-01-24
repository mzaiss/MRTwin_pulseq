clear all; close all;

addpath  ../SequenceSIM/3rdParty/pulseq-master/matlab/
addpath optfnc

% forward Fourier transform
fftfull =  @(x) ifftshift(fftn(fftshift(x)))/sqrt(numel(x));
ifftfull =  @(x) ifftshift(ifftn(fftshift(x)))*sqrt(numel(x));

% NRMSE error function
e = @(utrue,u) 100*norm(u(:)-utrue(:))/norm(utrue(:));

%% Check numerical vs. analytical derivatives

T = 4;       % number of time points in readout                                                                                                  
sz = [4, 6]; % image size (Nx Ny)                                                                                                       

% set gradient spatial forms
rampX = pi*linspace(-1,1,sz(1) + 1);
rampX = rampX(1:end-1);
rampX = repmat(rampX.', [1, sz(2)]);

% set gradient spatial forms
rampY = pi*linspace(-1,1,sz(2) + 1);
rampY = rampY(1:end-1);
rampY = repmat(rampY, [sz(1), 1]);

% initialize gradients (X/Y directions)
g = rand(T,2); g = g(:);

% initialize complex-valued magnetization image
m = rand(sz(1),sz(2)) + 1i*rand(sz(1),sz(2));

use_tanh_fieldcap = 1;                                                          % otherwise put L2 penalty on the field controlled by lambda
lambda = 0*1e-2;

% pack the parameters for the gradient function
args = {m, rampX, rampY,sz,lambda,use_tanh_fieldcap};
[phi,dg_ana] = phi_grad_readout2d(g(:),args{:}); % compute loss and analytical derivatives

% compute numerical derivatives
h = 1e-4;
dg = zeros(size(g)); dphi_h = zeros(size(g));
for i=1:numel(g)
  dg(i) = 1; dphi_h(i) = phi_grad_readout2d(g+h*dg,args{:})-phi;
  dg(i) = 0;
  
  if mod(i,100) == 0
    dx_num = dphi_h/h;
    fprintf('deriv-err=%1.3f%%\n',e(dx_num(1:i),dg_ana(1:i)))
  end
end
dx_num = dphi_h/h; clear dphi_h i dx

fprintf('deriv-err=%1.3f%%\n',e(dx_num,dg_ana(:)))

% compare analytical and numerical gradients
[dg_ana(:), dx_num]


%% do full optimization
close all;

% params
NRep = 16;                                                                                                           % number of repetitions
sz = [16,16];                                                                                                                   % image size
T = 24;                                                                                           % set the number of time points in readout
nmb_rand_restarts = 5;                                                                      % number of restarts with random initializations
do_versbose = 0;                                                                         % additionally show learned Fourier basis functions

% regularization parameters
use_tanh_fieldcap = 1;                                                                     % limit the effective field to sz*[-1..1]/2 range
lambda = 0*1e-6;                                                                                               % put L2 penalty on the field

gtruth_m = load('../../data/phantom.mat'); gtruth_m = gtruth_m.phantom;
gtruth_m = imresize(gtruth_m,sz);  % resize to something managable
 
%gtruth_m = fftfull(gtruth_m); gtruth_m(8:end,:) = 0; gtruth_m = ifftfull(gtruth_m);        % set some part of kspace to zero just for a test
                                                                     
% set the optimizer
p = struct();
nMVM = 200;  % number of optimization iterations
p.length = -nMVM;
p.method = 'LBFGS';

% set gradient spatial forms
rampX = pi*linspace(-1,1,sz(1) + 1);
rampX = rampX(1:end-1);
rampX = repmat(rampX.', [1, sz(2)]);

rampY = pi*linspace(-1,1,sz(2) + 1);
rampY = rampY(1:end-1);
rampY = repmat(rampY, [sz(1), 1]);

% initialize reconstructed image
reco_m = zeros(sz);
all_grad = cell(NRep,1);                                                                         % learned gradients at all repetition steps

for rep = 1:NRep
  
  minerr = 1e8;
  bestgrad = 0;
  
  % compute the current error to ground-truth
  error_m = gtruth_m - reshape(reco_m,sz);
  
  for rnd_restart = 1:nmb_rand_restarts
    
    % initialize gradients
    %g = zeros(T,2); g = g(:);                                                                       % initialize the gradients to all zeros
    %g = zeros(T,2); g(:,1) = rand(T,1) - 0.5; g = g(:);
    g = rand(T,2) - 0.5; g = g(:);                                                                                % good for random restarts

    % do optimization for g of E(g), loss --> (||error_m - E.T*E*error_m||^2 + lambda*||cumsum(g)||^2) 
    args = {error_m, rampX, rampY,sz,lambda,use_tanh_fieldcap};
    [g,phi] = minimize(g(:),'phi_grad_readout2d',p,args{:});
    
    % select the gradients with the lowest loss achieved
    phi = phi(end);
    if phi < minerr
      bestgrad = g;
      minerr = phi;
    end
  end
 
  % forward pass to compute the prediction, field and gradients
  [~,~,reco_current,E,field,grads] = phi_grad_readout2d(bestgrad(:),args{:});
  all_grad{rep} = bestgrad;
  figure(1), plot(grads); title(['learned gradients at repetition ', num2str(rep), ' blue - grad X, orange - grad Y']); xlabel('time'); ylabel('gradient strength (au)');
  
  % update the current reconstruction
  reco_m = reco_m + reshape(reco_current,sz);

  figure(2),
    subplot(2,2,1), imagesc(abs(reshape(error_m,sz))); title(['repetition ',num2str(rep),' : target to predict']);
    subplot(2,2,2), imagesc(abs(reshape(reco_current,sz))); title(['repetition ',num2str(rep),' :prediction']);
    subplot(2,2,3), imagesc(abs(reco_m)); title(['repetition ',num2str(rep),' : reconstruction, error=',num2str(e(gtruth_m(:),reco_m(:)))]);
    subplot(2,2,4), imagesc(abs(gtruth_m)); title('global target to predict (actual image)');
    
  E = reshape(E,[],sz(1),sz(2));
  
  if do_versbose
    figure(10),
      for t = 1:T
        basis_func = fftshift(squeeze(E(t,:,:)));
        subplot(4,6,t), imagesc(fftshift(angle(basis_func)));
      end
  end

  % plot actual sampled kspace locations  
  figure(5)
  c = ones(T,1)*rep;                                                                                                % color code repetitions
    hold on; scatter(field(:,2), field(:,1),[],c); hold off; axis([-8,8,-8,8]); title('kspace sampled locations'); xlabel('readout direction'); ylabel('phase encode direction');
      
  pause
end

return

%% plug learned gradients into the sequence constructor

seqFilename = 'seq/learned_grad.seq';

SeqOpts.resolution = sz;                                                                                                       % matrix size
SeqOpts.FOV = sz;
SeqOpts.TE = 10e-3;
SeqOpts.TR = 3000e-3;
SeqOpts.FlipAngle = pi/2;

% init sequence and system
seq = mr.Sequence();
sys = mr.opts();

% rf pulse
rf = mr.makeBlockPulse(SeqOpts.FlipAngle,'Duration',1e-3);

%gradients
Nx = SeqOpts.resolution(1); Ny = SeqOpts.resolution(2);

deltak=1/SeqOpts.FOV(1);
dt=1e-3;
riseTime = 5e-16; % use the same rise times for all gradients, so we can neglect them
adc = mr.makeAdc(1,'Duration',dt,'Delay',riseTime);

% TR delay (probably computet wrong..)
delayTR=ceil((SeqOpts.TR)/seq.gradRasterTime)*seq.gradRasterTime;

gradXevent=mr.makeTrapezoid('x','FlatArea',-24*deltak,'FlatTime',dt-2*riseTime,'RiseTime', riseTime);
gradYevent=mr.makeTrapezoid('y','FlatArea',24*deltak,'FlatTime',dt-2*riseTime,'RiseTime', riseTime);

% put blocks together
for rep=1:NRep
    seq.addBlock(rf);
    
    for kx = 1:T
      learned_grads = reshape(all_grad{rep},[],2);
      
      gradXevent.amplitude=23.1481*learned_grads(kx,1);
      gradYevent.amplitude=23.1481*learned_grads(kx,2);

      seq.addBlock(gradXevent,gradYevent,adc);
    end
    
    seq.addBlock(mr.makeDelay(delayTR))
end

%write sequence
seq.write(seqFilename);


























