
clear all; close all;

addpath  ../SequenceSIM
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
NRep = 32;                                                                                                           % number of repetitions
sz = [32,32];                                                                                                                   % image size
T = sz(2);                                                                                        % set the number of time points in readout
nmb_rand_restarts = 1;                                                                      % number of restarts with random initializations
do_versbose = 0;                                                                         % additionally show learned Fourier basis functions

% regularization parameters
use_tanh_grad_moms_cap = 1;                                                            % limit the effective grad_moms to sz*[-1..1]/2 range
lambda = 0*1e-6;                                                                                           % put L2 penalty on the grad_moms

gtruth_m = load('../../data/phantom.mat'); gtruth_m = gtruth_m.phantom;
gtruth_m = phantom(sz(1));
%gtruth_m = imread('cameraman.tif'); gtruth_m = single(gtruth_m)/255;
gtruth_m = imresize(gtruth_m,sz);  % resize to something managable

gtruth_m = abs(gtruth_m);
 
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
%rampX = fftshift(rampX(1:end-1),2);
rampX = repmat(rampX.', [1, sz(2)]);

rampY = pi*linspace(-1,1,sz(2) + 1);
rampY = rampY(1:end-1);
%rampY = fftshift(rampY(1:end-1),2);
rampY = repmat(rampY, [sz(1), 1]);

minerr = 1e8;

% initialize reconstructed image
all_grad = cell(NRep,1);                                                                         % learned gradients at all repetition steps

for rnd_restart = 1:nmb_rand_restarts

  % initialize gradients
  %g = zeros(T,2); g = g(:);                                                                       % initialize the gradients to all zeros
  %g = zeros(NRep,T,2); g(:,:,1) = rand(NRep,T,1) - 0.5; g = g(:);
  g = 1.0*(rand(NRep,T,2) - 0.5); g = g(:); 
  
  p.length = -500;

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

savedbestgrad = bestgrad;


%%
close all

grad_moms = zeros(NRep,T,2);
grad_moms(:,:,1) = repmat(linspace(-sz(1)/2,sz(1)/2-1,sz(1)),[sz(1),1,1]);
grad_moms(:,:,2) = repmat(linspace(-sz(2)/2,sz(2)/2-1,sz(2)).',[1 sz(2),1]);

%grad_moms = grad_moms + 0.5*(rand(size(grad_moms))-0.5);
%grad_moms(:,:,1) = grad_moms(:,:,1) + 10;
alpha = 10*pi/180; Rmat = [cos(alpha), -sin(alpha); sin(alpha), cos(alpha)];

for i = 1:NRep
  for j = 1:T
    grad_moms(i,j,:) = Rmat*squeeze(grad_moms(i,j,:));
  end
end

bestgrad = diff(cat(2,zeros(sz(1),1,2),grad_moms),[],2); use_tanh_grad_moms_cap = 0;

g = bestgrad;

lambda = 0;

args = {gtruth_m,rampX,rampY,adc_mask,sz,NRep,lambda,use_tanh_grad_moms_cap,0};
[~,~,reco_current,E_all_rep,grad_moms_output,grads] = phi_grad_allrep_readout2d(g(:),args{:});

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

learned_grads_all = grads;



return

%% plug learned gradients into the sequence constructor
% close all
NRep = 2;
seqFilename = 'seq/learned_grad.seq';

SeqOpts.resolution = sz;                                                                                                       % matrix size
SeqOpts.FOV = 220e-3;
SeqOpts.TE = 10e-3;
SeqOpts.TR = 10000e-3;
SeqOpts.FlipAngle = pi/2;

% init sequence and system
seq = mr.Sequence();
sys = mr.opts();

% rf pulse
rf = mr.makeBlockPulse(SeqOpts.FlipAngle,'Duration',1e-3);

%gradients
Nx = SeqOpts.resolution(1); Ny = SeqOpts.resolution(2);


deltak=1/SeqOpts.FOV;
dt=1e-3;
riseTime = 10e-36; % use the same rise times for all gradients, so we can neglect them
adc = mr.makeAdc(1,'Duration',dt,'Delay',riseTime);

% TR delay (probably computet wrong..)
delayTR=ceil((SeqOpts.TR)/seq.gradRasterTime)*seq.gradRasterTime;

% learned_grads are implicitly gradients of 1s length
%grad_moms = learned_grads * 1; % 1s

clear gradXevent gradYevent

gradXevent=mr.makeTrapezoid('x','FlatArea',-deltak,'FlatTime',dt-2*riseTime,'RiseTime', riseTime);
gradYevent=mr.makeTrapezoid('y','FlatArea',deltak,'FlatTime',dt-2*riseTime,'RiseTime', riseTime);

amplitude = abs(gradXevent.amplitude);

NRep = sz(2);


% put blocks together
for rep=1:NRep
    seq.addBlock(rf);
    
     learned_grads = reshape(learned_grads_all{rep},[],2);
%     grad_moms = cumsum(learned_grads,1) * 1; % 1s
    
    learned_grads = learned_grads * 1; % 1s
    
        
    figure(11), scatter(learned_grads(:,1),learned_grads(:,2)); title('gradmoms'); hold on;
    cum_grad_moms = cumsum([learned_grads],1) * 1; % 1s
    figure(111), scatter(cum_grad_moms(:,2)+sz(1)/2,cum_grad_moms(:,1)+sz(1)/2,'Displayname','from BlochSim'); title('cumgradmoms'); hold on;
    
    for kx = 1:T
      
      %gradXevent=mr.makeTrapezoid('x','FlatArea',flatArea*grad_moms(kx,1),'FlatTime',dt-2*riseTime,'RiseTime', riseTime);
      %gradYevent=mr.makeTrapezoid('y','FlatArea',flatArea*grad_moms(kx,2),'FlatTime',dt-2*riseTime,'RiseTime', riseTime);
      
      gradXevent.amplitude=-1*learned_grads(kx,1)*amplitude;
      gradYevent.amplitude=-1*learned_grads(kx,2)*amplitude;

      seq.addBlock(gradXevent,gradYevent,adc);
    end
    
    seq.addBlock(mr.makeDelay(delayTR))
end

%write sequence
seq.write(seqFilename);

seq.plot();


%% CONVENTIONAL

% close all

PD = phantom(sz(1));
%PD = abs(gtruth_m);

PD(PD<0) = 0;
T1 = 1e6*PD*2; T1(:) = 1;
T2 = 1e6*PD*2; T2(:) = 1;
InVol = double(cat(3,PD,T1,T2));

numSpins = 1;
[kList, gradMoms] = RunMRIzeroBlochSimulationNSpins(InVol, seqFilename, 1);

resolution = sz(1);

kList(isnan(kList))=0;
gradMomsScaled = (gradMoms+0.5)*resolution;  % calculate grad moms to FoV

[X,Y] = meshgrid(1:resolution);

kReco = griddata(gradMomsScaled(1,:),gradMomsScaled(2,:),real(kList),X,Y) +1j*griddata(gradMomsScaled(1,:),gradMomsScaled(2,:),imag(kList),X,Y) ;
% kReco = griddata(field(:,1),field(:,2),real(kList),X,Y) +1j*griddata(field(:,1),field(:,2),imag(kList),X,Y) ;
kReco(isnan(kReco))=0;

figure, subplot(2,2,1), imagesc(abs(fft2(fftshift(kReco))));
subplot(2,2,2), scatter(gradMoms(1,:),gradMoms(2,:)); title('gradMoms');
figure(111), scatter(gradMomsScaled(2,:),gradMomsScaled(1,:),'.'); title('gradMomsScaled: circles from cumsum(g), dots from BlochSim');

%{
figure(2), subplot(3,2,1), imagesc(abs(kReco),[0 200]);
subplot(3,2,2), imagesc(abs(kRef),[0 200]);
subplot(3,2,4), imagesc(abs(ifft2(fftshift(kRef))),[0 2]);

subplot(3,2,5), plot(gradMomsScaled(1,:),gradMomsScaled(2,:));
subplot(3,2,6),scatter3(gradMomsScaled(1,:),gradMomsScaled(2,:),real(kList))
drawnow;
clc   

%}

%%  E'E
PD1 = phantom(sz(1));

PD = abs(gtruth_m);

PD(PD<0) = 0;
T1 = 1e6*PD*2; T1(:) = 1;
T2 = 1e6*PD*2; T2(:) = 2;
InVol = double(cat(3,PD,T1,T2));

%InVol = permute(InVol,[2,1,3]);
%InVol = flipud(fliplr(InVol));

%InVol = fftshift(fftshift(InVol,1),2);

numSpins = 11;

[kList, gradMoms] = RunMRIzeroBlochSimulationNSpins(InVol, seqFilename, numSpins);

%kList = reshape(kList, [NRep, T]);
kList = reshape(kList, [T, NRep]);

reco = 0;
PD1 = eye(sz(1));
for rep = 1:NRep
  
  E = E_all_rep{rep};
  E = reshape(E, T, []);
  
  y = kList(:,rep);
  %y = kList(rep,:).';
  
  %y = E*gtruth_m(:);
  %y = E*PD1(:);
  
  reco = reco + E'*y;
  
end

reco = reshape(reco,sz);
reco = fftshift(reco);

figure(1), imagesc(abs(reco));


%% TEST pulseq and learngrads
clear grad_moms
%  plug learned gradients into the sequence constructor
% close all

sz = [32,32];   
seqFilename = 'seq/learned_grad_test.seq';

SeqOpts.resolution = sz;                                                                                                       % matrix size
SeqOpts.FOV = 220e-3;
SeqOpts.TE = 10e-3;
SeqOpts.TR = 1000000e-3;
SeqOpts.FlipAngle = pi/2;

% init sequence and system
seq = mr.Sequence();
sys = mr.opts();

% rf pulse
rf = mr.makeBlockPulse(SeqOpts.FlipAngle,'Duration',1e-3);

%gradients
Nx = SeqOpts.resolution(1); Ny = SeqOpts.resolution(2);

deltak=1/SeqOpts.FOV;
dt=1e-3;
riseTime = 10e-36; % use the same rise times for all gradients, so we can neglect them
adc = mr.makeAdc(1,'Duration',dt,'Delay',riseTime);

% TR delay (probably computet wrong..)
delayTR=ceil((SeqOpts.TR)/seq.gradRasterTime)*seq.gradRasterTime;

% learned_grads are implicitly gradients of 1s length
%grad_moms = learned_grads * 1; % 1s

clear gradXevent gradYevent

gradXevent=mr.makeTrapezoid('x','FlatArea',-deltak,'FlatTime',dt-2*riseTime,'RiseTime', riseTime);
gradYevent=mr.makeTrapezoid('y','FlatArea',deltak,'FlatTime',dt-2*riseTime,'RiseTime', riseTime);

amplitude = abs(gradXevent.amplitude);

% put blocks together
for rep=1:NRep
    seq.addBlock(rf);
    
%     learned_grads = reshape(all_grad{rep},[],2);
%     grad_moms = cumsum(learned_grads,1) * 1; % 1s
    
%     learned_grads=learned_grads * 1; % 1s
    
    grad_moms(:,1) = linspace(-sz(1)/2,sz(1)/2,sz(1));
    grad_moms(:,2) = ones(sz(1),1)*(rep-sz(1)/2);
    learned_grads = diff(cat(1,[0,0],grad_moms),1);
        
    figure(11), scatter(learned_grads(:,1),learned_grads(:,2)); title('gradmoms'); hold on;
    cum_grad_moms = cumsum([learned_grads],1) * 1; % 1s
    figure(111), scatter(cum_grad_moms(:,2)+sz(1)/2,cum_grad_moms(:,1)+sz(1)/2,'Displayname','from BlochSim'); title('cumgradmoms'); hold on;
    
    for kx = 1:T
      
      %gradXevent=mr.makeTrapezoid('x','FlatArea',flatArea*grad_moms(kx,1),'FlatTime',dt-2*riseTime,'RiseTime', riseTime);
      %gradYevent=mr.makeTrapezoid('y','FlatArea',flatArea*grad_moms(kx,2),'FlatTime',dt-2*riseTime,'RiseTime', riseTime);
      
      gradXevent.amplitude=learned_grads(kx,1)*amplitude;
      gradYevent.amplitude=learned_grads(kx,2)*amplitude;

      seq.addBlock(gradXevent,gradYevent,adc);
    end
    
    seq.addBlock(mr.makeDelay(delayTR))
end

%write sequence
seq.write(seqFilename);

seq.plot();

PD = phantom(sz(1));

PD(PD<0) = 0;
T1 = 1e6*PD*2; T1(:) = 1;
T2 = 1e6*PD*2; T2(:) = 100;
InVol = double(cat(3,PD,T1,T2));

  tic
        [kList, gradMoms] = RunMRIzeroBlochSimulationNSpins(InVol, seqFilename,1);
        toc
        
        kList(isnan(kList))=0;
        gradMomsScaled = (gradMoms+0.5)*sz(1);  % calculate grad moms to FoV
        
        [Y,X] = meshgrid(1:sz(1));
%         [Xq,Yq] = meshgrid(1:0.5:resolution);
%          kRefInterp = interp2(X,Y,kRef,gradMomsEnd(1), gradMomsEnd(2));
%         kRefInterp2 = interp2(X,Y,kRef,Xq, Yq);
               
        kReco = griddata(gradMomsScaled(1,:),gradMomsScaled(2,:),real(kList),X,Y) -1j*griddata(gradMomsScaled(1,:),gradMomsScaled(2,:),imag(kList),X,Y) ;
        kReco(isnan(kReco))=0;

        figure(), subplot(3,2,1), imagesc(abs(kReco),[0 200]);
               subplot(3,2,3), imagesc(abs(fft2(fftshift(kReco))),[0 500]); 
        subplot(3,2,5), plot(gradMomsScaled(1,:),gradMomsScaled(2,:));
        subplot(3,2,6),scatter3(gradMomsScaled(1,:),gradMomsScaled(2,:),real(kList))
       



























