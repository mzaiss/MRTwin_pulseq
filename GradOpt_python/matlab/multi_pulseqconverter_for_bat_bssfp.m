
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


ni = 30;

scanner_dict_target = load([seq_dir,'/','scanner_dict_tgt.mat']);

try
scanner_dict = load([seq_dir,'/','all_iter.mat']);   
niter = size(scanner_dict.flips,1);
sz = double(scanner_dict_target.sz);
T = scanner_dict.T;
NRep = scanner_dict.NRep;
catch
niter=0;
warning('No all_iter found, hust create Target seq');
sz = double(scanner_dict_target.sz);
T = size(scanner_dict_target.flips,1);
NRep = size(scanner_dict_target.flips,2);
end
    
disp(niter);

%%

k = 0;

idxarray_exported_itersteps = [1:150,160:10:niter];
idxarray_exported_itersteps = [1:150,160:10:niter];
idxarray_exported_itersteps = [1:20,30:10:niter];
idxarray_exported_itersteps = [1:50, 52:2:100, 110:10:niter];
% idxarray_exported_itersteps = [1:2:100, 100:10:150, 160:20:1140];
idxarray_exported_itersteps = [1:30 35:5:340 350:50:niter];
idxarray_exported_itersteps = [1:2:100 110:10:350 400:100:2000];

% idxarray_exported_itersteps = [niter-10:niter];
 idxarray_exported_itersteps = [1:niter];
 idxarray_exported_itersteps=[ 1
           2
           3
           4
           5
           6
           7
           8
           9
          10
          11
          12
          13
          14
          15
          16
          17
          18
          19
          20
          21
          22
          23
          24
          25
          26
          27
          28
          29
          30
          35
          40
          45
          50
          55
          60
          65
          70
          75
          80
          85
          90
          95
         100
         105
         110
         115
         120
         125
         130
         135
         140
         145
         150
         155
         160
         165
         170
         175
         180
         185
         190
         195
         200
         205
         210
         215
         220
         225
         230
         235
         240
         245
         250
         255
         260
         265
         270
         275
         280
         285
         290
         295
         300
         305
         310
         315
         320
         325
         330
         335
         340
         350
         400
         450
         500
         550
         600
         650
         700
         750
         800
         850
         900
         950
        1000
        1050
        1100
        1150
        1200
        1250]';
    idxarray_exported_itersteps=idxarray_exported_itersteps(90:end);
idxarray_exported_itersteps= idxarray_exported_itersteps(idxarray_exported_itersteps<=niter); % catch to high iteration numbers

for ni =  [0 idxarray_exported_itersteps] % add target seq in the beginning
    
    idx = double(ni);
    %print(idx);
    
    % plug learned gradients into the sequence constructor
    % close all
    seq_fn = [seq_dir,'/','seqiter',num2str(k),'.seq'];
    
    
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
    % read gradient
    if idx==0   % added traget sequence in the beginning
        flips = double(squeeze(scanner_dict_target.flips(:,:,:)));
        event_times = double(squeeze(scanner_dict_target.event_times(:,:)));
        gradmoms = double(squeeze(scanner_dict_target.grad_moms(:,:,:)))*deltak;  % that brings the gradmoms to the k-space unit of deltak =1/FoV
        
    else
        flips = double(squeeze(scanner_dict.flips(idx,:,:,:)));
        event_times = double(squeeze(scanner_dict.event_times(idx,:,:)));
        gradmoms = double(squeeze(scanner_dict.grad_moms(idx,:,:,:)))*deltak;  % that brings the gradmoms to the k-space unit of deltak =1/FoV
    end
    
nonsel = 0;

    % put blocks together
    for rep=1:NRep
        
        % first two extra events T(1:2)
        % first
        idx_T=1; % T(1)
        
        if flips(1,1,1) ==0  % take care of seqs with 0 flipangles
            flips(1,1,1) = 10^-2;
        end            
        RFdur=0;    
        if abs(flips(idx_T,rep,1)) > 1e-8
            use = 'excitation';
            if nonsel
            % block pulse non-selective
                sliceThickness=200e-3;
                rf = mr.makeBlockPulse(flips(idx_T,rep,1),'Duration',0.8*1e-3,'PhaseOffset',flips(idx_T,rep,2), 'use',use);
                seq.addBlock(rf);
                RFdur= 0.8*1e-3;
            else
            % alternatively slice selective:
                sliceThickness=5e-3;     % slice
                [rf, gz,gzr] = mr.makeSincPulse(single(flips(idx_T,rep,1)),'Duration',1e-3,'PhaseOffset',flips(idx_T,rep,2),'SliceThickness',sliceThickness,'apodization',0.5,'timeBwProduct',4,'system',sys,'use',use);
                seq.addBlock(rf,gz);
                seq.addBlock(gzr);  
                RFdur= gz.riseTime+gz.flatTime+gz.fallTime+ (gzr.riseTime+gzr.flatTime+gzr.fallTime);
            end
        end
        (event_times(idx_T,rep)-RFdur); % this is the actual timing from spyder
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
        
%         adc = mr.makeAdc(numel(idx_T),'Duration',gx.flatTime,'Delay',gx.riseTime);
        
        %update revinder for gxgy ramp times, from second event
        gxPre = mr.makeTrapezoid('x','Area',gradmom_revinder(1)-gx.amplitude*gx.riseTime/2,'Duration',eventtime_revinder,'system',sys);
        gyPre = mr.makeTrapezoid('y','Area',gradmom_revinder(2)-gy.amplitude*gy.riseTime/2,'Duration',eventtime_revinder,'system',sys);
        seq.addBlock(gxPre,gyPre); % add updated rewinder event from second event, including the full event time
        
        seq.addBlock(gx,gy,adc);  % add ADC grad event
        
        % second last extra event  T(end)  % adjusted also for fallramps of ADC
        idx_T=size(gradmoms,1)-1; % T(2)
        gxPost = mr.makeTrapezoid('x','Area',gradmoms(idx_T,rep,1)-gx.amplitude*gx.fallTime/2,'Duration',event_times(idx_T,rep),'system',sys);
        gyPost = mr.makeTrapezoid('y','Area',gradmoms(idx_T,rep,2)-gy.amplitude*gy.fallTime/2,'Duration',event_times(idx_T,rep),'system',sys);
        if nonsel
            seq.addBlock(gxPost,gyPost); 
        else
            seq.addBlock(gxPost,gyPost,gzr); % added gzr rewinder for balancing
        end
        
        %  last extra event  T(end)
        idx_T=size(gradmoms,1); % T(2)
        seq.addBlock(mr.makeDelay(event_times(idx_T,rep)))
        
        
    end
    
    seq.setDefinition('FOV', [SeqOpts.FOV SeqOpts.FOV sliceThickness]*1e3);

    %write sequence
    seq.write(seq_fn);
   % seq.writeBinary(seq_fn);
    
    pause(2);
    fprintf('%d of %d\n',k,numel(idxarray_exported_itersteps));
    if k==0
        seq.plot(); subplot(3,2,2); title('Target sequence');
        figure, colormap(jet);
        subplot(2,3,1), imagesc(scanner_dict_target.flips(:,:,1)'*180/pi); title('Flips'); colormap(gca,jet(fix(max(max(scanner_dict_target.flips(:,:,1)'*180/pi))))); colorbar; 
        subplot(2,3,4), imagesc(scanner_dict_target.flips(:,:,2)'*180/pi); title('Phases'); colormap(gca,jet(fix(max(max(scanner_dict_target.flips(:,:,2)'*180/pi))/20))); colorbar;
        subplot(2,3,2), imagesc(scanner_dict_target.event_times'); title('delays');colorbar
        subplot(2,3,3), imagesc(scanner_dict_target.grad_moms(:,:,1)');         title('gradmomx');colorbar
        subplot(2,3,6), imagesc(scanner_dict_target.grad_moms(:,:,2)');          title('gradmomy');colorbar
        
        [ktraj_adc, ktraj, t_excitation, t_refocusing] = seq.calculateKspace();
        figure; plot(ktraj'); % plot the entire k-space trajectory
%         figure; plot(ktraj(1,:),ktraj(2,:),'c',...
%             ktraj_adc(1,:),ktraj_adc(2,:),'g.'); % a 2D plot
        figure; plot3(ktraj(1,:),ktraj(2,:),ktraj(3,:),'c',...
            ktraj_adc(1,:),ktraj_adc(2,:),ktraj_adc(3,:),'g.'); % a 3D plot
        axis('equal'); % enforce aspect ratio for the correct trajectory display

    end
    %subplot(3,2,1), title(experiment_id,'Interpreter','none');
    k = k + 1;    
end

if niter>0
save([seq_dir,'/export_protocol.mat'],'idxarray_exported_itersteps','experiment_id');
end
