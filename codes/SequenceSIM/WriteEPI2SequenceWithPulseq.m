function seq = WriteEPI2SequenceWithPulseq(SeqOpts, seqFilename)

% check inputs
if nargin < 2
    error('Too less input arguments. Please provide SeqOpts and filename');
end

if ~isfield(SeqOpts, 'resolution')
    error('SeqOpts.resolution is a required input!');
end

% set standard parameters if not provided
if ~isfield(SeqOpts, 'FOV')
    SeqOpts.FOV = SeqOpts.resolution;
end

if ~isfield(SeqOpts, 'TE')
    SeqOpts.TE = 10e-3;
end

if ~isfield(SeqOpts, 'TR')
    SeqOpts.TR = 1;
end

if ~isfield(SeqOpts, 'FlipAngle')
    SeqOpts.FlipAngle = pi/2;
end


% init sequence and system
seq = mr.Sequence();
sys = mr.opts();

% rf pulse
rf = mr.makeBlockPulse(SeqOpts.FlipAngle,'Duration',3.2e-3);

%gradients
Nx = SeqOpts.resolution;
Ny = SeqOpts.resolution;
deltak=1/SeqOpts.FOV;
riseTime = 5e-16; % use the same rise times for all gradients, so we can neglect them
gx = mr.makeTrapezoid('x','FlatArea',deltak,'FlatTime',3.2e-3/Nx,'RiseTime', riseTime);

% needed because in current BlochSimulator.cpp kx is increased before taking
% first sample -> kx shift needs to be compensated before next line
gxCorrectionBlip = mr.makeTrapezoid('x','FlatArea',deltak,'FlatTime',3.2e-3,'RiseTime', riseTime);

adc = mr.makeAdc(1,'Duration',gx.flatTime,'Delay',gx.riseTime);
gxPre = mr.makeTrapezoid('x','Area',-Nx*deltak/2,'Duration',3.2e-3, 'RiseTime', riseTime);
gyPre = mr.makeTrapezoid('y','Area',-Ny*deltak/2,'Duration',3.2e-3, 'RiseTime', riseTime);

gyBlip = mr.makeTrapezoid('y','Area',deltak,'Duration',3.2e-3,'RiseTime', riseTime);


% put blocks together
seq.addBlock(rf);
seq.addBlock(gxPre,gyPre);

for i=1:Ny
    seq.addBlock(gyBlip);
    seq.addBlock(mr.makeDelay(1e-3));
    
    for jj=1:Nx
        seq.addBlock(gx,adc);
    end
    
    seq.addBlock(gxCorrectionBlip);
    
    gx.amplitude = -gx.amplitude;
    gxCorrectionBlip.amplitude = -gxCorrectionBlip.amplitude;
    
end


%write sequence
seq.write(seqFilename);

