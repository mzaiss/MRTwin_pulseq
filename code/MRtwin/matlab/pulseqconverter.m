
clear all; close all;

if isunix
  mrizero_git_dir = '/is/ei/aloktyus/git/mrizero_tueb';
else
  mrizero_git_dir = 'D:/root/ZAISS_LABLOG/LOG_MPI/27_MRI_zero/mrizero_tueb';
end

addpath([ mrizero_git_dir,'/codes/SequenceSIM']);
addpath([ mrizero_git_dir,'/codes/SequenceSIM/3rdParty/pulseq-master/matlab/']);

seq_dir = [mrizero_git_dir '/codes/GradOpt_python/out/'];
%experiment_id = 'RARE_FA_OPT_fixrep1_90';
experiment_id = 'FLASH_target';
% experiment_id = 'GRE_base';
% experiment_id = 'RARE_FA_OPT_fixrep1_90_adjflipgrad';
%  experiment_id = 'RARE_FA_OPT_fixrep1_90_adjflipgrad_spoiled';

%param_dict = load([seq_dir,'/',experiment_id,'/','param_dict.mat']);
%spins_dict = load([host_dir,'/',experiment_id,'/','spins_dict.mat']);
scanner_dict = load([seq_dir,'/',experiment_id,'/','scanner_dict.mat']);

sz = double(scanner_dict.sz);

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
button = questdlg('Generate for Scanner or Simulation?','MaxSlewrate check','Scanner','Simulation','Simulation');
if strcmp(button,'Scanner') maxSlew=140; else maxSlew=140*1000000000; end; sprintf('sys.maxSlew %d',maxSlew)
% had to slow down ramps and increase adc_duration to avoid stimulation
sys = mr.opts('MaxGrad',36,'GradUnit','mT/m',...
    'MaxSlew',maxSlew,'SlewUnit','T/m/s',...
    'rfRingdownTime', 20e-6, 'rfDeadTime', 100e-6, ...
    'adcDeadTime', 20e-6);

sys2 = mr.opts('MaxGrad',36,'GradUnit','mT/m',...
    'MaxSlew',maxSlew/3,'SlewUnit','T/m/s',...
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

gxPre = mr.makeTrapezoid('x','Area',sz(1)/SeqOpts.FOV,'Duration',scanner_dict.event_times(1,1),'system',sys);
     
% put blocks together
for rep=1:NRep

    gradmoms = double(scanner_dict.grad_moms)*deltak;  % that brings the gradmoms to the k-space unit of deltak =1/FoV

    % first two extra events T(1:2)
    % first
      idx_T=1; % T(1)
          
      if abs(scanner_dict.flips(idx_T,rep,1)) > 1e-8
        use = 'excitation';
        rf = mr.makeBlockPulse(scanner_dict.flips(idx_T,rep,1),'Duration',0.8*1e-3,'PhaseOffset',scanner_dict.flips(idx_T,rep,2), 'use',use);
        seq.addBlock(rf);
      end
%       seq.addBlock(mr.makeDelay(scanner_dict.event_times(idx_T,rep)))
      gxPre = mr.makeTrapezoid('x','Area',gradmoms(idx_T,rep,1),'Duration',scanner_dict.event_times(idx_T,rep),'system',sys);
      gyPre = mr.makeTrapezoid('y','Area',gradmoms(idx_T,rep,2),'Duration',scanner_dict.event_times(idx_T,rep),'system',sys);
      seq.addBlock(gxPre,gyPre);
      % alternatively slice selective:
        %[rf, gz, gzr] = makeSincPulse(scanner_dict.flips(idx_T,rep,1))
        % see writeHASTE.m      
      
    % second      
        idx_T=2; % T(2)
        use = 'refocusing';
        if abs(scanner_dict.flips(idx_T,rep,1)) > 1e-8
          rf = mr.makeBlockPulse(scanner_dict.flips(idx_T,rep,1),'Duration',0.8*1e-3,'PhaseOffset',scanner_dict.flips(idx_T,rep,2), 'use',use);
          seq.addBlock(rf);
        end
        seq.addBlock(mr.makeDelay(scanner_dict.event_times(idx_T,rep)))      
        
        gxPre = mr.makeTrapezoid('x','Area',gradmoms(idx_T,rep,1),'Duration',scanner_dict.event_times(idx_T,rep),'system',sys);
        gyPre = mr.makeTrapezoid('y','Area',gradmoms(idx_T,rep,2),'Duration',scanner_dict.event_times(idx_T,rep),'system',sys2);
        seq.addBlock(gxPre,gyPre);
      
    % line acquisition T(3:end-1)
        idx_T=3:size(gradmoms,1)-1; % T(2)
        dur=sum(scanner_dict.event_times(3:end-1,rep));
        gx = mr.makeTrapezoid('x','Area',sum(gradmoms(idx_T,rep,1),1),'Duration',dur,'system',sys);
        adc = mr.makeAdc(numel(idx_T),'Duration',dur-2*gx.riseTime-2*gx.fallTime,'Delay',2*gx.riseTime);
      
        seq.addBlock(gx,adc);
      
    % last extra event  T(end)
        idx_T=size(gradmoms,1); % T(2)
        gxPost = mr.makeTrapezoid('x','Area',gradmoms(idx_T,rep,1),'Duration',scanner_dict.event_times(idx_T,rep),'system',sys);
        gyPost = mr.makeTrapezoid('y','Area',gradmoms(idx_T,rep,2),'Duration',scanner_dict.event_times(idx_T,rep),'system',sys2);
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
figure; plot(ktraj(1,:),ktraj(2,:),'c',...
             ktraj_adc(1,:),ktraj_adc(2,:),'g.'); % a 2D plot
axis('equal'); % enforce aspect ratio for the correct trajectory display



%% CONVENTIONAL RECO

cum_grad_moms = cumsum([double(scanner_dict.grad_moms)],1);
cum_grad_moms = cum_grad_moms(find(scanner_dict.adc_mask),:,:);

% seqFilename='tse.seq'
seqFilename=seq_fn;

sz=double(scanner_dict.sz)
% close all

PD = phantom(sz(1));
%PD = abs(gtruth_m);

PD(PD<0) = 0;
T1 = 1e6*PD*2; T1(:) = 1;
T2 = 1e6*PD*2; T2(:) = 1;
InVol = double(cat(3,PD,T1,T2));

numSpins = 1;
[kList, kloc] = RunMRIzeroBlochSimulationNSpins(InVol, seqFilename, 1);
kloc = reshape(kloc,2,sz(1),sz(2));
kloc(1,:,:) = kloc(1,:,:) 
% kloc=ktraj_adc/max(ktraj_adc(:))*0.5;
%gradMoms(1,:,3) = gradMoms(1,:,3) + 1;
%gradMoms(1,:,13) = gradMoms(1,:,13) + 1;
%gradMoms = reshape(permute(cum_grad_moms,[3,1,2]),[2,prod(sz)]);

resolution = sz(1);

kList(isnan(kList))=0;
klocScaled = (kloc+0.5)*resolution;  % calculate grad moms to FoV

[X,Y] = meshgrid(1:resolution);
kReco = griddata(klocScaled(1,:),klocScaled(2,:),real(kList),X,Y) +1j*griddata(klocScaled(1,:),klocScaled(2,:),imag(kList),X,Y) ;
% kReco = griddata(field(:,1),field(:,2),real(kList),X,Y) +1j*griddata(field(:,1),field(:,2),imag(kList),X,Y) ;
kReco=reshape(kList,sz)
kReco(isnan(kReco))=0;

figure, subplot(2,2,1), imagesc(fftshift(abs(fft2(fftshift(kReco)))));
subplot(2,2,2), scatter(kloc(1,:),kloc(2,:)); title('k location (cumsum gradMoms)');
figure(111), scatter(klocScaled(2,:),klocScaled(1,:),'.'); title('k loc scaled: circles from cumsum(g), dots from BlochSim');

spectrum = reshape(kList,sz);

%% ADJOINT RECO
adjoint_mtx = scanner_dict.adjoint_mtx;
adjoint_mtx = adjoint_mtx(:,1:2,find(scanner_dict.adc_mask),:,1:2);
adjoint_mtx = adjoint_mtx(:,1,:,:,1) + 1i*adjoint_mtx(:,1,:,:,2);
adjoint_mtx = reshape(adjoint_mtx,[prod(sz),prod(sz)]);

signal_opt = squeeze(scanner_dict.signal(:,find(scanner_dict.adc_mask),:,1) + 1i*scanner_dict.signal(:,find(scanner_dict.adc_mask),:,2));

kList = reshape(kList,sz);

spectrum = reshape(kList,sz);
%spectrum = signal_opt;
spectrum = spectrum(:);

reco = adjoint_mtx*spectrum;
reco = reshape(reco,sz);

%{
adjoint_mtx = reshape(adjoint_mtx,[prod(sz),sz(1),sz(2)]);

reco = 0;
for rep = 1:NRep
  y = kList(:,rep);
  reco = reco + adjoint_mtx(:,:,rep)*y;
end

reco = reshape(reco,sz);
%}

figure(1),
  imagesc(abs(reco));

  
  
  

%% FIRST approach of full individual gradmoms (WIP)

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














