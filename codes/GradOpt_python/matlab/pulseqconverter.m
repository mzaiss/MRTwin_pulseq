
clear all; close all;

mrizero_git_dir = 'D:/root/ZAISS_LABLOG/LOG_MPI/27_MRI_zero/mrizero_tueb';

addpath([ mrizero_git_dir,'/codes/SequenceSIM']);
addpath([ mrizero_git_dir,'/codes/SequenceSIM/3rdParty/pulseq-master/matlab/']);

seq_dir = [mrizero_git_dir '/codes/GradOpt_python/out/'];
experiment_id = 'RARE_FA_OPT_fixrep1_90';
experiment_id = 'RARE_baseline';

%param_dict = load([seq_dir,'/',experiment_id,'/','param_dict.mat']);
%spins_dict = load([host_dir,'/',experiment_id,'/','spins_dict.mat']);
scanner_dict = load([seq_dir,'/',experiment_id,'/','scanner_dict.mat']);

% gradient tranform
grad_moms = scanner_dict.grad_moms;

if 0  % only when gradients were optimized, make sure gradmoms are between -res/2 res/2
    fmax = scanner_dict.sz / 2;                                                                                % cap the grad_moms to [-1..1]*sz/2
    for i = 1:2
      grad_moms(:,:,i) = fmax(i)*tanh(grad_moms(:,:,i));                                                                        % soft threshold
      %grad_moms(abs(grad_moms(:,i)) > fmax(i),i) = sign(grad_moms(abs(grad_moms(:,i)) > fmax(i),i))*fmax(i);  % hard threshold, this part is nondifferentiable
    end
end

figure,
colormap 'jet'
subplot(2,3,1), imagesc(scanner_dict.flips(:,:,1)'); title('Flips'); colorbar
subplot(2,3,4), imagesc(scanner_dict.flips(:,:,2)'); title('Phases');colorbar
subplot(2,3,2), imagesc(scanner_dict.event_times'); title('delays');colorbar
subplot(2,3,3), imagesc(grad_moms(:,:,1)');         title('gradmomx');colorbar
subplot(2,3,6), imagesc(grad_moms(:,:,2)');          title('gradmomy');colorbar
set(gcf,'OuterPosition',[431         379        1040         513])
%% plug learned gradients into the sequence constructor
% close all
seq_fn = [seq_dir,'/',experiment_id,'/','pulseq.seq'];

SeqOpts.resolution = double(scanner_dict.sz);                                                                                            % matrix size
SeqOpts.FOV = 220e-3;
SeqOpts.TE = 10e-3;          % fix
SeqOpts.TR = 10000e-3;       % fix
SeqOpts.FlipAngle = pi/2;    % fix

% set system limits
% had to slow down ramps and increase adc_duration to avoid stimulation
sys = mr.opts('MaxGrad',36,'GradUnit','mT/m',...
    'MaxSlew',140*1000000,'SlewUnit','T/m/s',...
    'rfRingdownTime', 20e-6, 'rfDeadTime', 100e-6, ...
    'adcDeadTime', 20e-6);


%gradients
Nx = SeqOpts.resolution(1); Ny = SeqOpts.resolution(2);

% ok, there are two ways to do it now:
% APPROACH A: each gradient event is an individual block, with ramp up and down,
% this would be necessary for free gradient moms

% APPROACH B: we assume at least line acquisition in each repetition, thus the
% 16 actions within one repetition are played out as one gradient event
% with 16 samples

%% APPROACH B: line read approach
% we have to calculate the actually necessary gradients from the grad moms
% this is easy when using the AREA of the pulseqgrads  and gradmoms*deltak

% init sequence and system
seq = mr.Sequence();
% ADC duration (controls TR/TE)
adc_dur=2560; %us

% Define other gradients and ADC events
deltak=1/SeqOpts.FOV;
% read gradient

T = size(scanner_dict.grad_moms,1);
NRep = size(scanner_dict.grad_moms,2);

% put blocks together
for rep=1:NRep

    gradmoms = double(scanner_dict.grad_moms)*deltak;  % that brings the gradmoms to the k-space unit of deltak =1/FoV

    % first two extra events T(1:2)
    % first
      idx_T=1; % T(1)
      
      if rep == 1
          use = 'excitation';
      else
          use = 'refocusing';
      end
      
      rf = mr.makeBlockPulse(scanner_dict.flips(idx_T,rep,1),'Duration',0.8*1e-3,'PhaseOffset',scanner_dict.flips(idx_T,rep,2), 'use',use);
      seq.addBlock(rf);
      seq.addBlock(mr.makeDelay(scanner_dict.event_times(rep,idx_T)))
      % alternatively slice selective:
        %[rf, gz, gzr] = makeSincPulse(scanner_dict.flips(idx_T,rep,1))
        % see writeHASTE.m      
      
    % second      
        idx_T=2; % T(2)
        gxPre = mr.makeTrapezoid('x','Area',gradmoms(idx_T,rep,1),'Duration',scanner_dict.event_times(idx_T,rep),'system',sys);
        gyPre = mr.makeTrapezoid('y','Area',gradmoms(idx_T,rep,2),'Duration',scanner_dict.event_times(idx_T,rep),'system',sys);
      seq.addBlock(gxPre,gyPre);
      
    % line acquisition T(3:end-1)
        idx_T=3:size(gradmoms,1)-1; % T(2)
        dur=sum(scanner_dict.event_times(3:end-1,rep));
        gx = mr.makeTrapezoid('x','Area',sum(gradmoms(idx_T,rep,1),1),'Duration',dur,'system',sys);
        adc = mr.makeAdc(numel(idx_T),'Duration',dur,'Delay',gx.riseTime);
      seq.addBlock(gx,adc);
      
    % last extra event  T(end)
        idx_T=size(gradmoms,1); % T(2)
        gxPost = mr.makeTrapezoid('x','Area',gradmoms(idx_T,rep,1),'Duration',scanner_dict.event_times(idx_T,rep),'system',sys);
        gyPost = mr.makeTrapezoid('y','Area',gradmoms(idx_T,rep,2),'Duration',scanner_dict.event_times(idx_T,rep),'system',sys);
      seq.addBlock(gxPost,gyPost);

end

%write sequence
seq.write(seq_fn);

seq.plot();
subplot(3,2,1), title(experiment_id,'Interpreter','none');


return

%% new single-function call for trajectory calculation
[ktraj_adc, ktraj, t_excitation, t_refocusing] = seq.calculateKspace();

% plot k-spaces

figure; plot(ktraj'); % plot the entire k-space trajectory
figure; plot(ktraj(1,:),ktraj(2,:),'b',...
             ktraj_adc(1,:),ktraj_adc(2,:),'r.'); % a 2D plot
axis('equal'); % enforce aspect ratio for the correct trajectory display




%% FIRST approach of full individual gradmoms

% Define other gradients and ADC events
deltak=1/SeqOpts.FOV;
gx = mr.makeTrapezoid('x','FlatArea',Nx*deltak,'FlatTime',adc_dur*1e-6,'system',sys);
adc = mr.makeAdc(Nx,'Duration',gx.flatTime,'Delay',gx.riseTime,'system',sys);
gxPre = mr.makeTrapezoid('x','Area',-gx.area/2,'system',sys);
phaseAreas = ((0:Ny-1)-Ny/2)*deltak;

dt=1e-3;
riseTime = 10e-36; % use the same rise times for all gradients, so we can neglect them
adc = mr.makeAdc(1,'Duration',dt,'Delay',riseTime);

% TR delay (probably computet wrong..)
delayTR=ceil((SeqOpts.TR)/seq.gradRasterTime)*seq.gradRasterTime;

clear gradXevent gradYevent

gradXevent=mr.makeTrapezoid('x','FlatArea',-deltak,'FlatTime',dt-2*riseTime,'RiseTime', riseTime);
gradYevent=mr.makeTrapezoid('y','FlatArea',deltak,'FlatTime',dt-2*riseTime,'RiseTime', riseTime);

amplitude = abs(gradXevent.amplitude);

T = param_dict.T;
NRep = sz(2);

% put blocks together
for rep=1:NRep
    seq.addBlock(rf);
    
    learned_grads = scanner_dict.grads;
    learned_grads = learned_grads * 1; % 1s
    
    
    % TANH correct ?????
    
    figure(11), scatter(learned_grads(:,rep,1),learned_grads(:,rep,2)); title('gradmoms'); hold on;
    cum_grad_moms = cumsum([learned_grads],2) * 1; % 1s
    figure(111), scatter(cum_grad_moms(:,rep,2)+sz(1)/2,cum_grad_moms(:,rep,1)+sz(1)/2,'Displayname','from BlochSim'); title('cumgradmoms'); hold on;
    
    for kx = 1:T
      
      gradXevent.amplitude=1*learned_grads(kx,rep,1)*amplitude;
      gradYevent.amplitude=1*learned_grads(kx,rep,2)*amplitude;

      seq.addBlock(gradXevent,gradYevent,adc);
    end
    
    seq.addBlock(mr.makeDelay(delayTR))
end

%write sequence
seq.write(seq_fn);

seq.plot();


return

%% CONVENTIONAL

% seqFilename='tse.seq'
seqFilename=seq_fn;

sz=[24 24]
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
%kReco=reshape(kList,sz)
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

%PD(17:32,:) = 0; PD(:,17:32) = 0;
%PD(1:16,:) = 0; PD(:,1:16) = 0;

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
       



























