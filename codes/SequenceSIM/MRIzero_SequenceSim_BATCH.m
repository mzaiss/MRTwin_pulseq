% MRIzero BATCH


%% create Input image (N,N,3)
% (:,:,1) -> PD
% (:,:,2) -> T1
% (:,:,3) -> T2

resolution = 48; % 100x100 take runs ~12s on a single core
PD = phantom(resolution);
PD(PD<0) = 0;
T1 = PD*2;
T2 = PD*0.35;
InVol = cat(3,PD,T1,T2);
numSpins = 100;


%% Sequence Parameters

SeqOpts.resolution = resolution;
SeqOpts.FOV = resolution;
SeqOpts.TE = 10e-3;
SeqOpts.TR = 3000e-3;
SeqOpts.FlipAngle = pi/2;
seqFilename = fullfile(pwd, 'gre.seq');


sequence = WriteGRESequenceWithPulseq(SeqOpts, seqFilename);
%sequence.plot();

%% run simulation
tic;
[kList, gradients] = RunMRIzeroBlochSimulationNSpins(InVol, seqFilename, numSpins);
toc;


%% plot results

% plot kspace
kspace = reshape(kList,[resolution resolution]);
figure;
%original image
subplot(2,4,1);
imshow(PD);
subplot(2,4,2);
Y = fftshift(fft2(fftshift(PD)));
imagesc(abs(Y));
subplot(2,4,3);
imagesc(real(Y));
subplot(2,4,4);
imagesc(imag(Y));

% aquired signal
subplot(2,4,5);
imshow(abs(ifft2(fftshift(kspace)))./numSpins);
subplot(2,4,6);
imagesc(abs(kspace));
subplot(2,4,7);
imagesc(real(kspace));
subplot(2,4,8);
imagesc(imag(kspace));

%plot trajectory
figure,plot(gradients(1,:),gradients(2,:), '-o');
