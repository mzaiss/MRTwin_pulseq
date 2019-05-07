
clear all; close all;

if isunix
    mrizero_git_dir = '/is/ei/aloktyus/git/mrizero_tueb';
    seq_dir = [mrizero_git_dir '/codes/GradOpt_python/out/'];
    %experiment_id = 'RARE_FA_OPT_fixrep1_90';
    experiment_id = 'e21_tgtRARE_tskRARE_32_linear';
    % experiment_id = 'GRE_base';
    % experiment_id = 'RARE_FA_OPT_fixrep1_90_adjflipgrad';
    %  experiment_id = 'RARE_FA_OPT_fixrep1_90_adjflipgrad_spoiled';
    seq_dir=[seq_dir,'/',experiment_id,'/'];
else
    mrizero_git_dir = 'D:/root/ZAISS_LABLOG/LOG_MPI/27_MRI_zero/mrizero_tueb';
    seq_dir = uigetdir('\\mrz3t\Upload\CEST_seq\pulseq_zero\sequences', 'Select a sequence folder');
    out=regexp(seq_dir,'\','split');
    experiment_id=out{end};
end

addpath([ mrizero_git_dir,'/codes/SequenceSIM']);
addpath([ mrizero_git_dir,'/codes/SequenceSIM/3rdParty/pulseq-master/matlab/']);



%param_dict = load([seq_dir,'/',experiment_id,'/','param_dict.mat']);
%spins_dict = load([host_dir,'/',experiment_id,'/','spins_dict.mat']);
scanner_dict_ = load([seq_dir,'/','scanner_dict_tgt.mat']);

sz = double(scanner_dict_.sz);

flips = double(squeeze(scanner_dict_.flips(:,:,:)));
event_times = double(squeeze(scanner_dict_.event_times(:,:)));
gradmoms = double(squeeze(scanner_dict_.grad_moms(:,:,:)));  % that brings the gradmoms to the k-space unit of deltak =1/FoV


figure,
colormap 'jet'
subplot(2,3,1), imagesc(flips(:,:,1)'); title('Flips'); colorbar
subplot(2,3,4), imagesc(flips(:,:,2)'); title('Phases');colorbar
subplot(2,3,2), imagesc(event_times'); title('delays');colorbar
subplot(2,3,3), imagesc(gradmoms(:,:,1)');         title('gradmomx');colorbar
subplot(2,3,6), imagesc(gradmoms(:,:,2)');          title('gradmomy');colorbar
set(gcf,'OuterPosition',[431         379        1040         513])
%% plug learned gradients into the sequence constructor
% close all
seq_fn = [seq_dir,'/','pulseq.seq'];

SeqOpts.resolution = double(scanner_dict_.sz);                                                                                            % matrix size
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
seq = mr.Sequence(sys);
% ADC duration (controls TR/TE)
adc_dur=2560; %us

% Define other gradients and ADC events
deltak=1/SeqOpts.FOV;
% read gradient

T = size(gradmoms,1);
NRep = size(gradmoms,2);

gradmoms = double(gradmoms)*deltak;  % that brings the gradmoms to the k-space unit of deltak =1/FoV

% put blocks together
for rep=1:NRep


    % first two extra events T(1:2)
    % first
      idx_T=1; % T(1)
      
      RFdur=0;
      if abs(flips(idx_T,rep,1)) > 1e-8
        use = 'excitation';
        RFdur=1*1e-3;
        sliceThickness=200e-3;
        rf = mr.makeBlockPulse(flips(idx_T,rep,1),'Duration',RFdur,'PhaseOffset',flips(idx_T,rep,2), 'use',use);
        seq.addBlock(rf);     
        gxPre90 = mr.makeTrapezoid('x','Area',gradmoms(idx_T,rep,1),'Duration',event_times(idx_T,rep)-RFdur,'system',sys);
        seq.addBlock(gxPre90);  % this is the revinder between 90 and first 180
      else
        seq.addBlock(mr.makeDelay(event_times(idx_T,rep))) % 
      end
      % alternatively slice selective:
        %[rf, gz, gzr] = makeSincPulse(scanner_dict.flips(idx_T,rep,1))
        % see writeHASTE.m      
      
    % second      
        idx_T=2; % T(2)
        use = 'refocusing';
        RFdur=0;
        if abs(flips(idx_T,rep,1)) > 1e-8
          RFdur=1*1e-3;
          rf = mr.makeBlockPulse(flips(idx_T,rep,1),'Duration',RFdur,'PhaseOffset',flips(idx_T,rep,2), 'use',use);
          seq.addBlock(rf);
        end
        seq.addBlock(mr.makeDelay(event_times(idx_T,rep)-RFdur)) % this ensures that the RF block is 2 ms as it must be defined in python, also dies when negative

        gradmom_revinder = squeeze(gradmoms(idx_T,rep,:));
        eventtime_revinder = squeeze(event_times(idx_T,rep));
             
    % line acquisition T(3:end-1)
        idx_T=3:size(gradmoms,1)-2; % T(2)
        dur=sum(event_times(idx_T,rep));
        gx = mr.makeTrapezoid('x','FlatArea',sum(gradmoms(idx_T,rep,1),1),'FlatTime',dur,'system',sys);
        adc = mr.makeAdc(numel(idx_T),'Duration',gx.flatTime,'Delay',gx.riseTime,'phaseOffset',rf.phaseOffset);
    
    %update revinder for gxgy ramp times, from second event
        gxPre = mr.makeTrapezoid('x','Area',gradmom_revinder(1)-gx.amplitude*gx.riseTime/2,'Duration',eventtime_revinder,'system',sys);
        gyPre = mr.makeTrapezoid('y','Area',gradmom_revinder(2),'Duration',eventtime_revinder,'system',sys);
        seq.addBlock(gxPre,gyPre);
        seq.addBlock(gx,adc);
      
    % second last extra event  T(end)
        idx_T=size(gradmoms,1)-1; % T(2)
        gxPost = mr.makeTrapezoid('x','Area',gradmoms(idx_T,rep,1)-gx.amplitude*gx.fallTime/2,'Duration',event_times(idx_T,rep),'system',sys);
        gyPost = mr.makeTrapezoid('y','Area',gradmoms(idx_T,rep,2),'Duration',event_times(idx_T,rep),'system',sys);
        seq.addBlock(gxPost,gyPost);
        
    %  last extra event  T(end)
        idx_T=size(gradmoms,1); % T(2)
        seq.addBlock(mr.makeDelay(event_times(idx_T,rep)))
     

end

seq.setDefinition('FOV', [SeqOpts.FOV SeqOpts.FOV sliceThickness]*1e3);

[ktraj_adc, ktraj, t_excitation, t_refocusing] = seq.calculateKspace();
figure; plot(ktraj'); % plot the entire k-space trajectory
figure; plot(ktraj(1,:),ktraj(2,:),'c',...
             ktraj_adc(1,:),ktraj_adc(2,:),'g.'); % a 2D plot
axis('equal'); % enforce aspect ratio for the correct trajectory display

%write sequence
seq.write(seq_fn);

seq.plot();
subplot(3,2,1), title(experiment_id,'Interpreter','none');


%% new single-function call for trajectory calculation
[ktraj_adc, ktraj, t_excitation, t_refocusing] = seq.calculateKspace();

% plot k-spaces

figure; plot(ktraj'); % plot the entire k-space trajectory
figure; plot(ktraj(1,:),ktraj(2,:),'c',...
             ktraj_adc(1,:),ktraj_adc(2,:),'g.'); % a 2D plot
axis('equal'); % enforce aspect ratio for the correct trajectory display

rep = seq.testReport;
fprintf([rep{:}]);


return


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














