
clear all; close all;

if isunix
    mrizero_git_dir = '/is/ei/aloktyus/git/mrizero_tueb';
    seq_dir = '/media/upload3t/CEST_seq/pulseq_zero/sequences/seq190403';
    experiment_id = 'e06_tgtGRE_tsk_GRE_no_grad_16_1kspins_lr0.1_onlyPE_50iter';
    seq_dir =[seq_dir '/' experiment_id];
else
    mrizero_git_dir = 'D:/root/ZAISS_LABLOG/LOG_MPI/27_MRI_zero/mrizero_tueb';
    seq_dir = uigetdir('\\mrz3t\Upload\CEST_seq\pulseq_zero\sequences', 'Select a sequence folder');
    out=regexp(seq_dir,'\','split');
    experiment_id=out{end};
end

addpath([ mrizero_git_dir,'/codes/SequenceSIM']);
addpath([ mrizero_git_dir,'/codes/SequenceSIM/3rdParty/pulseq-master/matlab/']);


ni = 20;

scanner_dict_target = load([seq_dir,'/','scanner_dict_tgt.mat']);
sz = double(scanner_dict_target.sz);
NRep = sz(1);

niter = size(scanner_dict_target.flips,1);

disp(niter);

%%

k = 0;

idxarray_exported_itersteps = [1:150,160:10:niter];
idxarray_exported_itersteps = [1:150,160:10:niter];
idxarray_exported_itersteps = [1:20,30:10:niter];
idxarray_exported_itersteps = [1:50, 52:2:100, 110:10:niter];
% idxarray_exported_itersteps = [1:2:100, 100:10:150, 160:20:1140];
idxarray_exported_itersteps = [1:30 35:5:340 350:50:niter];
idxarray_exported_itersteps = [1:2:100 110:10:190 205:5:400 450:50:1000];

idxarray_exported_itersteps= idxarray_exported_itersteps(idxarray_exported_itersteps<=niter);


    
    
SeqOpts.resolution = double(sz);                                                                                            % matrix size
SeqOpts.FOV = 220e-3;
SeqOpts.TE = 10e-3;          % fix
SeqOpts.TR = 10000e-3;       % fix
SeqOpts.FlipAngle = pi/2;    % fix


% set system limits
button = 'Scanner';
if strcmp(button,'Scanner') maxSlew=140; else maxSlew=140*1000000000; end; 
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

    flips = double(squeeze(scanner_dict_target.flips(:,:,:)));
    event_times = double(squeeze(scanner_dict_target.event_times(:,:)));
    gradmoms = double(squeeze(scanner_dict_target.grad_moms(:,:,:)))*deltak;  % that brings the gradmoms to the k-space unit of deltak =1/FoV

seq_fn = [seq_dir,'/','multimeasbssfp_noADCgy.seq'];

maxmeas = 50;
        
   
for ni = 1:maxmeas % add target seq in the beginning
    
    % put blocks together
    for rep=1:NRep
        
        % first two extra events T(1:2)
        % first
        idx_T=1; % T(1)
        
        if abs(flips(idx_T,rep,1)) > 1e-8
            use = 'excitation';
            nonsel=1;
            if nonsel
                % block pulse non-selective
                sliceThickness=200e-3;
                rf = mr.makeBlockPulse(flips(idx_T,rep,1),'Duration',0.8*1e-3,'PhaseOffset',flips(idx_T,rep,2), 'use',use);
                seq.addBlock(rf);
                RFdur=0.8*1e-3;
            else
                % alternatively slice selective:
                sliceThickness=5e-3;     % slice
                [rf, gz,gzr] = mr.makeSincPulse(single(flips(idx_T,rep,1)),'Duration',1e-3,'PhaseOffset',flips(idx_T,rep,2), 'use',use,'SliceThickness',sliceThickness,'apodization',0.5,'timeBwProduct',4,'system',sys);
                seq.addBlock(rf,gz);
                seq.addBlock(gzr);
                RFdur= gz.riseTime+gz.flatTime+gz.fallTime+ gzr.riseTime+gzr.flatTime+gzr.fallTime;
            end
        end
        seq.addBlock(mr.makeDelay(0.002-RFdur)) % this ensures that the RF block is 2 ms as it must be defined in python, also dies when negative
        
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
        
        seq.addBlock(gx,adc);  % add ADC grad event
        
        % second last extra event  T(end)  % adjusted also for fallramps of ADC
        idx_T=size(gradmoms,1)-1; % T(2)

        gxPost = mr.makeTrapezoid('x','Area',gradmoms(idx_T,rep,1)-gx.amplitude*gx.fallTime/2,'Duration',event_times(idx_T,rep),'system',sys);

        gyPost = mr.makeTrapezoid('y','Area',gradmoms(idx_T,rep,2)-gy.amplitude*gy.fallTime/2,'Duration',event_times(idx_T,rep),'system',sys);
        seq.addBlock(gxPost,gyPost);
        %  last extra event  T(end)
        idx_T=size(gradmoms,1); % T(2)
        seq.addBlock(mr.makeDelay(event_times(idx_T,rep)))
        
        
    end
     
    fprintf('%d of %d\n',ni,maxmeas);
    if ni==1
        seq1=seq;    
        seq1.plot();
        figure,
        [ktraj_adc, ktraj, t_excitation, t_refocusing] = seq1.calculateKspace();
        figure; plot(ktraj'); % plot the entire k-space trajectory
        figure; plot(ktraj(1,:),ktraj(2,:),'c',...
                 ktraj_adc(1,:),ktraj_adc(2,:),'g.'); % a 2D plot
        axis('equal'); % enforce aspect ratio for the correct trajectory display
        drawnow;
    end
    %subplot(3,2,1), title(experiment_id,'Interpreter','none');
 
end

seq.setDefinition('FOV', [SeqOpts.FOV SeqOpts.FOV sliceThickness]*1e3);
%write sequence
seq.write(seq_fn);
% seq.writeBinary(seq_fn);
    
    
save([seq_dir,'/export_protocol.mat'],'idxarray_exported_itersteps','experiment_id');
