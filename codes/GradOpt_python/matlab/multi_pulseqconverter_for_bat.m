
clear all; close all;

if isunix
  mrizero_git_dir = '/is/ei/aloktyus/git/mrizero_tueb';
  seq_dir = '/media/upload3t/CEST_seq/pulseq_zero/sequences/seq190403';
else
  mrizero_git_dir = 'D:/root/ZAISS_LABLOG/LOG_MPI/27_MRI_zero/mrizero_tueb';
  seq_dir = 'K:\CEST_seq\pulseq_zero\sequences';
end


addpath([ mrizero_git_dir,'/codes/SequenceSIM']);
addpath([ mrizero_git_dir,'/codes/SequenceSIM/3rdParty/pulseq-master/matlab/']);


experiment_id = 'e06_tgtGRE_tsk_GRE_no_grad_16_1kspins_lr0.1_onlyPE_50iter';

ni = 30;

scanner_dict = load([seq_dir,'/',experiment_id,'/','all_iter.mat']);
sz = double(scanner_dict.sz);
T = scanner_dict.T;
NRep = scanner_dict.NRep;

niter = size(scanner_dict.flips,1);


%%

k = 1;

idxarray = [1:150,160:10:1840];
idxarray = [1:10:500];

for ni =  idxarray
  
  idx = double(ni);
  %print(idx);

  % plug learned gradients into the sequence constructor
  % close all
  seq_fn = [seq_dir,'/',experiment_id,'/','seqiter',num2str(k),'.seq'];
  k = k + 1;

  SeqOpts.resolution = double(sz);                                                                                            % matrix size
  SeqOpts.FOV = 220e-3;
  SeqOpts.TE = 10e-3;          % fix
  SeqOpts.TR = 10000e-3;       % fix
  SeqOpts.FlipAngle = pi/2;    % fix


  % set system limits
  button = 'Scanner';
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
  
  flips = double(squeeze(scanner_dict.flips(idx,:,:,:)));
  event_times = double(squeeze(scanner_dict.event_times(idx,:,:)));
  gradmoms = double(squeeze(scanner_dict.grad_moms(idx,:,:,:)))*deltak;  % that brings the gradmoms to the k-space unit of deltak =1/FoV

  % put blocks together
  for rep=1:NRep

      % first two extra events T(1:2)
      % first
        idx_T=1; % T(1)

        if abs(flips(idx_T,rep,1)) > 1e-8
          use = 'excitation';
          rf = mr.makeBlockPulse(flips(idx_T,rep,1),'Duration',0.8*1e-3,'PhaseOffset',flips(idx_T,rep,2), 'use',use);
          seq.addBlock(rf);
        end
  %       seq.addBlock(mr.makeDelay(scanner_dict.event_times(idx_T,rep)))
        gxPre = mr.makeTrapezoid('x','Area',gradmoms(idx_T,rep,1),'Duration',event_times(idx_T,rep),'system',sys);
        gyPre = mr.makeTrapezoid('y','Area',gradmoms(idx_T,rep,2),'Duration',event_times(idx_T,rep),'system',sys);
        seq.addBlock(gxPre,gyPre);
        % alternatively slice selective:
          %[rf, gz, gzr] = makeSincPulse(scanner_dict.flips(idx_T,rep,1))
          % see writeHASTE.m      

      % second      
          idx_T=2; % T(2)
          use = 'refocusing';
          if abs(flips(idx_T,rep,1)) > 1e-8
            rf = mr.makeBlockPulse(flips(idx_T,rep,1),'Duration',0.8*1e-3,'PhaseOffset',flips(idx_T,rep,2), 'use',use);
            seq.addBlock(rf);
          end
          seq.addBlock(mr.makeDelay(event_times(idx_T,rep)))      

          gxPre = mr.makeTrapezoid('x','Area',gradmoms(idx_T,rep,1),'Duration',event_times(idx_T,rep),'system',sys);
          gyPre = mr.makeTrapezoid('y','Area',gradmoms(idx_T,rep,2),'Duration',event_times(idx_T,rep),'system',sys2);
          seq.addBlock(gxPre,gyPre);

      % line acquisition T(3:end-1)
          idx_T=3:size(gradmoms,1)-2; % T(2)
          dur=sum(event_times(idx_T,rep));
          gx = mr.makeTrapezoid('x','Area',sum(gradmoms(idx_T,rep,1),1),'Duration',dur,'system',sys);
          adc = mr.makeAdc(numel(idx_T),'Duration',dur-2*gx.riseTime-2*gx.fallTime,'Delay',2*gx.riseTime,'phaseOffset',rf.phaseOffset);
          seq.addBlock(gx,adc);

      % second last extra event  T(end)
          idx_T=size(gradmoms,1)-1; % T(2)
          gxPost = mr.makeTrapezoid('x','Area',gradmoms(idx_T,rep,1),'Duration',event_times(idx_T,rep),'system',sys);
          gyPost = mr.makeTrapezoid('y','Area',gradmoms(idx_T,rep,2),'Duration',event_times(idx_T,rep),'system',sys2);
          seq.addBlock(gxPost,gyPost);
      %  last extra event  T(end)
          idx_T=size(gradmoms,1); % T(2)
          seq.addBlock(mr.makeDelay(event_times(idx_T,rep)))   


  end

  %write sequence
  seq.write(seq_fn);
  
  pause(2);

  %seq.plot();
  %subplot(3,2,1), title(experiment_id,'Interpreter','none');

end
