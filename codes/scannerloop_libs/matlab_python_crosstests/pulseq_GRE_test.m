
clear all;
close all;

if isunix
    mrizero_git_dir = '/is/ei/aloktyus/git/mrizero_tueb';
    seq_dir = '/is/ei/aloktyus/git/mrizero_tueb/codes/scannerloop_libs/matlab_python_crosstests/seq_and_data';
else
    mrizero_git_dir = 'D:/root/ZAISS_LABLOG/LOG_MPI/27_MRI_zero/mrizero_tueb';
    seq_dir = 'D:/root/ZAISS_LABLOG/LOG_MPI/27_MRI_zero/mrizero_tueb/codes/scannerloop_libs/matlab_python_crosstests/seq_and_data';
end

addpath([ mrizero_git_dir,'/codes/SequenceSIM']);
addpath([ mrizero_git_dir,'/codes/SequenceSIM/3rdParty/pulseq-master/matlab/']);


scanner_dict = load([seq_dir,'/scanner_dict_tgt.mat']);

sz = double(scanner_dict.sz);

% gradient tranform
grad_moms = scanner_dict.grad_moms;

figure,
subplot(2,3,1), imagesc(scanner_dict.flips(:,:,1)'*180/pi); title('Flips'); colormap(gca,jet(fix(max(max(scanner_dict.flips(:,:,1)'*180/pi))))); colorbar; 
subplot(2,3,4), imagesc(scanner_dict.flips(:,:,2)'*180/pi); title('Phases'); colormap(gca,jet(fix(max(max(scanner_dict.flips(:,:,2)'*180/pi))/20))); colorbar;
subplot(2,3,2), imagesc(scanner_dict.event_times'); title('delays');colorbar
subplot(2,3,3), imagesc(grad_moms(:,:,1)');         title('gradmomx');colorbar
subplot(2,3,6), imagesc(grad_moms(:,:,2)');          title('gradmomy');colorbar
set(gcf,'OuterPosition',[431         379        1040         513])
% plug learned gradients into the sequence constructor
% close all
seq_fn = [seq_dir,'/test_matlab.seq'];

SeqOpts.resolution = double(scanner_dict.sz);                                                                                            % matrix size
SeqOpts.FOV = 220e-3;
SeqOpts.TE = 10e-3;          % fix
SeqOpts.TR = 10000e-3;       % fix
SeqOpts.FlipAngle = pi/2;    % fix


% set system limits
% button = questdlg('Generate for Scanner or Simulation?','MaxSlewrate check','Scanner','Simulation','Simulation');
button= 'Scanner';
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

% APPROACH B: line read approach
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
flips = double(squeeze(scanner_dict.flips(:,:,:)));
event_times = double(squeeze(scanner_dict.event_times(:,:)));
gradmoms = double(squeeze(scanner_dict.grad_moms(:,:,:)))*deltak;  % that brings the gradmoms to the k-space unit of deltak =1/FoV


% put blocks together
for rep=1:NRep
    
    % first two extra events T(1:2)
    % first
    idx_T=1; % T(1)
    
    if abs(flips(idx_T,rep,1)) > 1e-8
        use = 'excitation';
%         sliceThickness=200*e-3;
%         rf = mr.makeBlockPulse(flips(idx_T,rep,1),'Duration',0.8*1e-3,'PhaseOffset',flips(idx_T,rep,2), 'use',use);
%         seq.addBlock(rf);
        % alternatively slice selective:
        sliceThickness=5e-3;     % slice
        [rf, gz,gzr] = mr.makeSincPulse(single(flips(idx_T,rep,1)),'Duration',0.6*1e-3,'SliceThickness',sliceThickness,'apodization',0.5,'timeBwProduct',4,'system',sys);
%         seq.addBlock(gzr);  % fully balanced Z
        seq.addBlock(rf,gz);
        seq.addBlock(gzr);
    end
    seq.addBlock(mr.makeDelay(event_times(idx_T,rep)))
%     gxPre = mr.makeTrapezoid('x','Area',gradmoms(idx_T,rep,1),'Duration',event_times(idx_T,rep),'system',sys);
%     gyPre = mr.makeTrapezoid('y','Area',gradmoms(idx_T,rep,2),'Duration',event_times(idx_T,rep),'system',sys);
%     seq.addBlock(gxPre,gyPre);

    
    % second  (rewinder)
    idx_T=2; % T(2)
    % calculated here, update in next event
    gradmom_revinder = squeeze(gradmoms(idx_T,rep,:));
    eventtime_revinder = squeeze(event_times(idx_T,rep));
    
    % line acquisition T(3:end-1)
    idx_T=3:size(gradmoms,1)-2; % T(2)
    dur=sum(event_times(idx_T,rep));
    
    gx = mr.makeTrapezoid('x','FlatArea',sum(gradmoms(idx_T,rep,1),1),'FlatTime',dur,'system',sys);
    gy = mr.makeTrapezoid('y','FlatArea',sum(gradmoms(idx_T,rep,2),1),'FlatTime',dur,'system',sys);
    adc = mr.makeAdc(numel(idx_T),'Duration',gx.flatTime,'Delay',gx.riseTime,'phaseOffset',rf.phaseOffset);
    
    %update revinder for gxgy ramp times, from second event
    gxPre = mr.makeTrapezoid('x','Area',gradmom_revinder(1)-gx.amplitude*gx.riseTime/2,'Duration',eventtime_revinder,'system',sys);
    gyPre = mr.makeTrapezoid('y','Area',gradmom_revinder(2)-gy.amplitude*gy.riseTime/2,'Duration',eventtime_revinder,'system',sys);
    seq.addBlock(gxPre,gyPre); % add updated rewinder event from second event, including the full event time
    
    seq.addBlock(gx,gy,adc);  % add ADC grad event
    
    % second last extra event  T(end)  % adjusted also for fallramps of ADC
    idx_T=size(gradmoms,1)-1; % T(2)
    gxPost = mr.makeTrapezoid('x','Area',gradmoms(idx_T,rep,1)-gx.amplitude*gx.fallTime/2,'Duration',event_times(idx_T,rep),'system',sys);
    gyPost = mr.makeTrapezoid('y','Area',gradmoms(idx_T,rep,2)-gy.amplitude*gy.fallTime/2,'Duration',event_times(idx_T,rep),'system',sys);
    seq.addBlock(gxPost,gyPost);
    %  last extra event  T(end)
    idx_T=size(gradmoms,1); % T(2)
    seq.addBlock(mr.makeDelay(event_times(idx_T,rep)))
    
    
end

seq.setDefinition('FOV', [SeqOpts.FOV SeqOpts.FOV sliceThickness]*1e3);

%write sequence
seq.write(seq_fn);

% seq.plot();
% subplot(3,2,1), title(experiment_id,'Interpreter','none');




%% new single-function call for trajectory calculation
[ktraj_adc, ktraj, t_excitation, t_refocusing] = seq.calculateKspace();

% plot k-spaces
ktraj_adc=ktraj_adc/deltak;
ktraj=ktraj/deltak;
figure; plot(ktraj'); % plot the entire k-space trajectory
figure(88); plot(ktraj(1,:),ktraj(2,:),'c',...
    ktraj_adc(1,:),ktraj_adc(2,:),'go'); hold on;  % a 2D plot
axis('equal'); % enforce aspect ratio for the correct trajectory display
legend({'k-pulseq','ADC-pulseq'});

% from SIM
grad_moms = squeeze(scanner_dict.grad_moms);
grad_moms =cat(1,zeros(1,NRep,2),grad_moms);
temp = squeeze(cumsum(grad_moms(:,:,1:2),1));
ktraj_adc_sim_x =temp(:,:,1);  ktraj_adc_sim_x =ktraj_adc_sim_x(:);
ktraj_adc_sim_y =temp(:,:,2);  ktraj_adc_sim_y =ktraj_adc_sim_y(:);

figure(88); plot(ktraj_adc_sim_x,ktraj_adc_sim_y,'bx-','DisplayName','k-sim'); hold on;% a 2D plot

%   ktraj_adc_sim = ktraj_adc_sim(3:end-2,:,:);
%   ktraj_adc_temp = reshape(permute(ktraj_adc_sim,[3,2,1]),2,[]);




