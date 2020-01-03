% change path to demoSeq
% cd X:\pulseq-master\matlab\demoSeq
% then run this script

writeTSE;

close all

% plot k-spaces  % new single-function call for trajectory calculation
[ktraj_adc, ktraj] = seq.calculateKspace();
figure; 
subplot(2,2,1), plot(ktraj'); title('before IO'); % plot the entire k-space trajectory 
subplot(2,2,2), plot(ktraj(1,:),ktraj(2,:),'b',ktraj_adc(1,:),ktraj_adc(2,:),'r.'); title('before IO'); % a 2D plot
axis('equal'); % enforce aspect ratio for the correct trajectory display
% write to and read from file

seq.write('tse.seq')
seq=mr.Sequence(system);
% seq.read('tse.seq')
seq.read('tse.seq','detectRFuse')  % try this instead

[ktraj_adc, ktraj] = seq.calculateKspace();
subplot(2,2,3), plot(ktraj'); title('after IO');% plot the entire k-space trajectory
subplot(2,2,4), plot(ktraj(1,:),ktraj(2,:),'b',ktraj_adc(1,:),ktraj_adc(2,:),'r.'); title('after IO'); % a 2D plot


