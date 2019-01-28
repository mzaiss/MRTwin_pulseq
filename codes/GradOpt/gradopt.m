% TODO: ADC masking

clear all; close all;

addpath ls 
addpath  ../SequenceSIM
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

adc_mask = ones(T,1); adc_mask(1:2) = 0;

% initialize gradients (X/Y directions)
g = rand(T,2); g = g(:);

% initialize complex-valued magnetization image
m = rand(sz(1),sz(2)) + 1i*rand(sz(1),sz(2));

use_tanh_grad_moms_cap = 1;                                                 % otherwise put L2 penalty on the grad_moms controlled by lambda
lambda = 0*1e-2;

% pack the parameters for the gradient function
args = {m,rampX,rampY,adc_mask,sz,lambda,use_tanh_grad_moms_cap};
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
T = 16;                                                                                           % set the number of time points in readout
nmb_rand_restarts = 5;                                                                      % number of restarts with random initializations
do_versbose = 0;                                                                         % additionally show learned Fourier basis functions

% regularization parameters
use_tanh_grad_moms_cap = 1;                                                            % limit the effective grad_moms to sz*[-1..1]/2 range
lambda = 0*1e-6;                                                                                           % put L2 penalty on the grad_moms

gtruth_m = load('../../data/phantom.mat'); gtruth_m = gtruth_m.phantom;
gtruth_m = imresize(gtruth_m,sz);  % resize to something managable
 
%gtruth_m = fftfull(gtruth_m); gtruth_m(8:end,:) = 0; gtruth_m = ifftfull(gtruth_m);       % set some part of kspace to zero just for a test
                                                                     
% set the optimizer
p = struct();
nMVM = 200;  % number of optimization iterations
p.length = -nMVM;
p.method = 'LBFGS';

% set ADC mask
adc_mask = ones(T,1); %adc_mask(1:6) = 0;

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

E_allrep = cell(NRep,1);

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
    args = {error_m,rampX,rampY,adc_mask,sz,lambda,use_tanh_grad_moms_cap};
    [g,phi] = minimize(g(:),'phi_grad_readout2d',p,args{:});
    
    % select the gradients with the lowest loss achieved
    phi = phi(end);
    if phi < minerr
      bestgrad = g;
      minerr = phi;
    end
  end
 
  % forward pass to compute the prediction, gradient moments and gradients
  [~,~,reco_current,E,grad_moms,grads] = phi_grad_readout2d(bestgrad(:),args{:});
  all_grad{rep} = grads;
  figure(1), plot(grads); title(['learned gradients at repetition ', num2str(rep), ' blue - grad X, orange - grad Y']); xlabel('time'); ylabel('gradient strength (au)');
  
  % update the current reconstruction
  reco_m = reco_m + reshape(reco_current,sz);

  figure(2),
    subplot(2,2,1), imagesc(abs(reshape(error_m,sz))); title(['repetition ',num2str(rep),' : target to predict']);
    subplot(2,2,2), imagesc(abs(reshape(reco_current,sz))); title(['repetition ',num2str(rep),' :prediction']);
    subplot(2,2,3), imagesc(abs(reco_m)); title(['repetition ',num2str(rep),' : reconstruction, error=',num2str(e(gtruth_m(:),reco_m(:)))]);
    subplot(2,2,4), imagesc(abs(gtruth_m)); title('global target to predict (actual image)');
    
  E = reshape(E,[],sz(1),sz(2));
  
  E_allrep{rep} = E;
  
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
    hold on; scatter(grad_moms(:,2), grad_moms(:,1),[],c); hold off; axis([-8,8,-8,8]); title('kspace sampled locations'); xlabel('readout direction'); ylabel('phase encode direction');
      
  %pause
end

return

%% plug learned gradients into the sequence constructor

NRep = 2;

seqFilename = 'seq/learned_grad.seq';

SeqOpts.resolution = sz;                                                                                                       % matrix size
SeqOpts.FOV = 220e-3;
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

deltak=1/SeqOpts.resolution(1);
dt=1e-3;
riseTime = 5e-5; % use the same rise times for all gradients, so we can neglect them
adc = mr.makeAdc(1,'Duration',dt,'Delay',riseTime);

% TR delay (probably computet wrong..)
delayTR=ceil((SeqOpts.TR)/seq.gradRasterTime)*seq.gradRasterTime;


% learned_grads are implicitly gradients of 1s length
%grad_moms = learned_grads * 1; % 1s

clear gradXevent gradYevent

gradXevent=mr.makeTrapezoid('x','FlatArea',-deltak,'FlatTime',dt-2*riseTime,'RiseTime', riseTime);
gradYevent=mr.makeTrapezoid('y','FlatArea',deltak,'FlatTime',dt-2*riseTime,'RiseTime', riseTime);

flatArea = abs(gradXevent.flatArea);

NRep = 16;

% put blocks together
for rep=1:NRep
    seq.addBlock(rf);
    
    learned_grads = reshape(all_grad{rep},[],2);
    grad_moms = cumsum(learned_grads,1) * 1; % 1s
    
    
    for kx = 1:T
      
      %gradXevent=mr.makeTrapezoid('x','FlatArea',flatArea*grad_moms(kx,1),'FlatTime',dt-2*riseTime,'RiseTime', riseTime);
      %gradYevent=mr.makeTrapezoid('y','FlatArea',flatArea*grad_moms(kx,2),'FlatTime',dt-2*riseTime,'RiseTime', riseTime);
      
      gradXevent.amplitude=23.1481*grad_moms(kx,1)/8 ;
      gradYevent.amplitude=23.1481*grad_moms(kx,2)/8;

      seq.addBlock(gradXevent,gradYevent,adc);
    end
    
    seq.addBlock(mr.makeDelay(delayTR))
end

%write sequence
seq.write(seqFilename);

seq.plot();

%%
PD = phantom(sz(1));

PD = abs(gtruth_m);

PD(PD<0) = 0;
T1 = 1e6*PD*2; T1(:) = 1e3;
T2 = 1e6*PD*2; T2(:) = 1e3;
InVol = double(cat(3,PD,T1,T2));

numSpins = 1;

[kList, gradMoms] = RunMRIzeroBlochSimulationNSpins(InVol, seqFilename, numSpins);


kList = reshape(kList, [NRep, T]);


reco = 0;

for rep = 1:NRep
  
  E = E_allrep{rep};
  E = reshape(E, T, []);
  
  y = kList(rep,:).';
  
  %y = E*gtruth_m(:);
  
  reco = reco + E'*y;
  
end

figure(1), imagesc(abs(reshape(reco,sz)));


%%

close all

PD = phantom(sz(1));

PD = abs(gtruth_m);

PD(PD<0) = 0;
T1 = 1e6*PD*2; T1(:) = 1e3;
T2 = 1e6*PD*2; T2(:) = 1e3;
InVol = double(cat(3,PD,T1,T2));

numSpins = 1;
[kList, gradMoms] = RunMRIzeroBlochSimulationNSpins(InVol, seqFilename, numSpins);

resolution = sz(1);



kList(isnan(kList))=0;
gradMomsScaled = (gradMoms+0.5)*resolution;  % calculate grad moms to FoV

[X,Y] = meshgrid(1:resolution);

kReco = griddata(gradMomsScaled(1,:),gradMomsScaled(2,:),real(kList),X,Y) +1j*griddata(gradMomsScaled(1,:),gradMomsScaled(2,:),imag(kList),X,Y) ;
kReco(isnan(kReco))=0;


figure, imagesc(abs(fft2(fftshift(kReco))));

figure, plot(gradMoms(1,:),gradMoms(2,:)); title('gradMoms');
figure, plot(gradMomsScaled(1,:),gradMomsScaled(2,:)); title('gradMomsScaled');

%{
figure(2), subplot(3,2,1), imagesc(abs(kReco),[0 200]);
subplot(3,2,2), imagesc(abs(kRef),[0 200]);
subplot(3,2,4), imagesc(abs(ifft2(fftshift(kRef))),[0 2]);

subplot(3,2,5), plot(gradMomsScaled(1,:),gradMomsScaled(2,:));
subplot(3,2,6),scatter3(gradMomsScaled(1,:),gradMomsScaled(2,:),real(kList))
drawnow;
clc   

%}














