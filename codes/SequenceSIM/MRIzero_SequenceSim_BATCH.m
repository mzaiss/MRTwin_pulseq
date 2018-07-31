% MRIzero BATCH


%% Scanner properties (no need to change something here)
gyro_ratio = 267.52; % gyromagnetic ratio in [rad/µT]
B0 = 9.4;            % B0 [T]
% max gradient strength at 9.4 is 60mT/m, we work with a 1 mm resolution for
% now and treat our magnetic field strength in uT
maxGradient = 60; % [µT/mm]
pixelSize   = 1;  % [mm]
% from here we can calculate the minimum gradient time for phase encodeing (rise times etc are neglected)
minGradTime =  pi / (maxGradient*gyro_ratio*pixelSize);


%% create Input image (N,N,3)
% (:,:,1) -> PD
% (:,:,2) -> T1
% (:,:,3) -> T2

resolution = 100; % 100x100 take runs ~12s on a single core
PD = phantom(resolution);
PD (PD<0) = 0;
T1 = PD*2;
T2 = PD*0.035;
InVol = cat(3,PD,T1,T2);


%% Sequence Parameters

% Flip angle
FA = pi/2;
% TR (not really TR, but the time to wait after the event block)
TR = 8; % enough time for relaxation


%noo need to change smth here
tpulse = 1e-3;
w1pulse = FA/tpulse;
% TE
TEstep = 2*minGradTime/resolution;



%% Base events
%           w1        phi        xGrad              yGrad         t             ADC
RT_Pulse = [w1pulse    0          0                 0             tpulse        0];
RT_GDep  = [0          0         -maxGradient       maxGradient   minGradTime   0];
RT_RO    = [0          0          maxGradient       0             TEstep        1];
RT_GRew  = [0          0         0                 -maxGradient   minGradTime   0];
RT_TR    = [0          0         0                  0             TR            0];

% calculate the different PE gradient strength
% so far, the mex function simulates a linear readout from the bottom to the top row
cY = ((1:resolution)-(resolution/2))*2;

%start with an empty block
BaseBlock = [];

for y=1:resolution
    
    % first event RF pulse
    BaseBlock = [BaseBlock; RT_Pulse];
    
    %calculate current gradient strength
    yGrad = cY(y)/resolution*maxGradient;
    RT_GDep(4) = yGrad;
    
    %2nd event dephasing in x and y
    BaseBlock = [BaseBlock; RT_GDep];
    
    % k space line (readout event in every point in kspace)
    for x = 1:resolution
        BaseBlock = [BaseBlock;RT_RO];
    end
    
    % phase rewind
    RT_GRew(4) = -yGrad;
    
    %relaxation
    if y<resolution
        BaseBlock = [BaseBlock;RT_GRew;RT_TR];
    end
end


%% run simulation
tic;
kspace = RunMRIzeroBlochSimulation(InVol,BaseBlock,B0);
toc;


%% plot results
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
imshow(abs(ifft2(fftshift(kspace))));
subplot(2,4,6);
imagesc(abs(kspace));
subplot(2,4,7);
imagesc(real(kspace));
subplot(2,4,8);
imagesc(imag(kspace));


