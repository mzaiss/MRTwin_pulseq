% MRIzero BATCH


%% create Input image (N,N,3)
% (:,:,1) -> PD
% (:,:,2) -> T1
% (:,:,3) -> T2

resolution = 24; % 100x100 take runs ~12s on a single core
PD = phantom(resolution);
NSpins=1;

% simple test image
% PD(:,:)=0;
% PD(resolution/2, resolution/2)=1;

PD(PD<0) = 0;
T1 = PD*4;
T2 = PD*1;
T2star = PD*1;

% T1 = (PD+phantom([0.5 0.2 0.3 -0.4 -0.4 45], resolution))*2;
% T2 = (PD+phantom([1 0.1 0.2 0.4 0.5 0], resolution))*0.35;

InVol = cat(3,PD,T1,T2);


%% Sequence Parameters

SeqOpts.resolution = resolution;
SeqOpts.FOV = 220e-3;
SeqOpts.TE = 15e-3;
SeqOpts.TR = 1000e-3;
SeqOpts.ETL = resolution;
SeqOpts.FlipAngle = 90*pi/180;
SeqOpts.FlipAngle1 = 90*pi/180;
SeqOpts.FlipAngle2 = pi;
SeqOpts.Order = 'increase'; % increase, center, center_in, (half)
filename = 'tse.seq';
seqFilename = fullfile(pwd, filename);


sequence = WriteTSESequenceWithPulseq(SeqOpts, seqFilename);
sequence.plot();
% sequence.sound();
%% run simulation
% seqFilename = 'epi.seq';
tic;
[kList, gradients] = RunMRIzeroBlochSimulationNSpins(InVol, seqFilename, NSpins);
toc;

%% reconstruction
%kspace = reshape(kList,[resolution resolution]);
kspace = kReorder(kList, gradients);

%% regridding
kspace = k_regrid(kList, gradients, resolution);

%% plot results
plotSimulationResult(PD, kspace./NSpins);

% corrections with kspace
% kspace2 = kspace;
% for ii = 2:2:size(kspace2,2)
% kspace2(:,ii) = flip(circshift(kspace(:,ii),1));
% end
% kspace2 = flip(kspace2,2);
% plotSimulationResult(PD, kspace2./NSpins);

% figure, imagesc(abs(nonuniformIDFT(kList, gradients, resolution))), title('nonuniformIDFT');

%% plot kspace-trajectory
plotKSpaceTrajectory(gradients, resolution, 0);

%% nonuniform IDFT
res_nuidft = nonuniformIDFT(kList, gradients, resolution);
figure
subplot(1,2,1), imagesc1t(abs(res_nuidft)), axis('image'), title('non-uniform IDFT');
subplot(1,2,2), imagesc1t(abs(PD)), axis('image'), title('REF (PD)');


%% comparison with pulseq trajectory
[ktraj_adc, ktraj, t_excitation, t_refocusing] = sequence.calculateKspace();

% figure; plot(ktraj'); % plot the entire k-space trajectory
figure; plot(ktraj(1,:),ktraj(2,:),'b'); % a 2D plot
hold;plot(ktraj_adc(1,:),ktraj_adc(2,:),'r.');

%% comparison of reco result with filtered original kspace
filterfunc1 = @(x,y) exp(-abs(y)/10); % exponential decay in PE direction
filterfunc2 = @(x,y) exp(-abs(y-resolution/2)/15) + exp(-abs(y+resolution/2)/15); % exponential decay in PE direction
filterfunc3 = @(x,y) exp(-(x.^2+y.^2)./(10)^2); % 2D gaussian
filterfunc4 = @(x,y) exp(-(sqrt(x.^2+y.^2))./15); % exp. decay with radius
[kFiltered, filter] = kApplyFilter(fftshift(fft2(fftshift(PD))), filterfunc4);
figure
subplot(2,2,1), mesh(filter), title('filter function');
subplot(2,2,4), imshow(abs(fftshift(ifft2(fftshift(kFiltered))))), title('filtered orig');
subplot(2,2,3), imshow(abs(ifft2(fftshift(kspace)))), title('reco');
subplot(2,2,2), imshow(abs(PD)), title('orig');
