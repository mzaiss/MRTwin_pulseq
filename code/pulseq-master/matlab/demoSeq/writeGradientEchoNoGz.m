seq=mr.Sequence();              % Create a new sequence object
fov=220e-3; Nx=42; Ny=42;     % Define FOV and resolution
alpha=5;                       % flip angle
sliceThickness=200e-3;            % slice
TE=[3.2]*1e-3;                % give a vector here to have multiple TEs (e.g. for field mapping)
TR=6.5e-3;                       % only a single value for now

% more in-depth parameters
rfSpoilingInc=117;              % RF spoiling increment

% set system limits
sys = mr.opts('MaxGrad', 28, 'GradUnit', 'mT/m', ...
    'MaxSlew', 1500000000, 'SlewUnit', 'T/m/s', 'rfRingdownTime', 20e-6, ...
    'rfDeadTime', 100e-6, 'adcDeadTime', 10e-6);

% Create alpha-degree slice selection pulse and gradient
[rf] = mr.makeBlockPulse(alpha*pi/180,'Duration',4e-3,'system',sys);

% Define other gradients and ADC events
deltak=1/fov;
gx = mr.makeTrapezoid('x','FlatArea',Nx*deltak,'FlatTime',6.4e-3,'system',sys);
adc = mr.makeAdc(Nx,'Duration',gx.flatTime,'Delay',gx.riseTime,'system',sys);
gxPre = mr.makeTrapezoid('x','Area',-gx.area/2,'Duration',2e-3,'system',sys);
%gzReph = mr.makeTrapezoid('z','Area',-gz.area/2,'Duration',2e-3,'system',sys);
phaseAreas = ((0:Ny-1)-Ny/2)*deltak;

% gradient spoiling
gxSpoil=mr.makeTrapezoid('x','Area',2*Nx*deltak,'system',sys);
gzSpoil=mr.makeTrapezoid('z','Area',4/sliceThickness,'system',sys);


% Calculate timing
delayTE=ceil((TE - mr.calcDuration(gxPre) -  ...
    - mr.calcDuration(gx)/2)/seq.gradRasterTime)*seq.gradRasterTime;
delayTR=ceil((TR - mr.calcDuration(gxPre) -  ...
    - mr.calcDuration(gx) - delayTE)/seq.gradRasterTime)*seq.gradRasterTime;


rf_phase=0;
rf_inc=0;

% Loop over phase encodes and define sequence blocks
for i=1:Ny
    for c=1:length(TE)
        rf.phaseOffset=rf_phase/180*pi;
        adc.phaseOffset=rf_phase/180*pi;
        rf_inc=mod(rf_inc+rfSpoilingInc, 360.0);
        rf_phase=mod(rf_phase+rf_inc, 360.0);
        %
        seq.addBlock(rf);
        gyPre = mr.makeTrapezoid('y','Area',phaseAreas(i),'Duration',2e-3,'system',sys);
        seq.addBlock(gxPre,gyPre);
        seq.addBlock(mr.makeDelay(delayTE(c)));
        seq.addBlock(gx,adc);
        gyPre.amplitude=-gyPre.amplitude;
        seq.addBlock(mr.makeDelay(delayTR(c)),gxSpoil,gyPre,gzSpoil)
    end
end


%% plot sequence and k-space diagrams

seq.plot();

% new single-function call for trajectory calculation
[ktraj_adc, ktraj, t_excitation, t_refocusing, t_adc] = seq.calculateKspace();

% plot k-spaces
time_axis=(1:(size(ktraj,2)))*sys.gradRasterTime;
figure; plot(time_axis, ktraj'); % plot the entire k-space trajectory
hold; plot(t_adc,ktraj_adc(1,:),'.'); % and sampling points on the kx-axis
figure,
plot(ktraj(1,:),ktraj(2,:),'b'); % a 2D plot
axis('equal'); % enforce aspect ratio for the correct trajectory display
hold; plot(ktraj_adc(1,:),ktraj_adc(2,:),'r.'); % plot the sampling points

%% check whether the timing of the sequence is correct
[ok, error_report]=seq.checkTiming;

if (ok)
    fprintf('Timing check passed successfully\n');
else
    fprintf('Timing check failed! Error listing follows:\n');
    fprintf([error_report{:}]);
    fprintf('\n');
end

%% prepare sequence export
%seq.setDefinition('FOV', [fov fov sliceThickness]*1e3);
seq.setDefinition('Name', 'gre');

seq.write('gre_nogz.seq')       % Write to pulseq file

%seq.install('siemens');
return

%% very optional slow step, but useful for testing during development e.g. for the real TE, TR or for staying within slewrate limits  

rep = seq.testReport;
fprintf([rep{:}]);

