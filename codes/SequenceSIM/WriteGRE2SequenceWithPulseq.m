function seq = WriteGRE2SequenceWithPulseq(SeqOpts, seqFilename)

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

if ~isfield(SeqOpts, 'Order')
    SeqOpts.Order = 'increase';
end

% init sequence and system
seq = mr.Sequence();
sys = mr.opts();

% rf pulse
rf = mr.makeBlockPulse(SeqOpts.FlipAngle,'Duration',1e-3);

%gradients
Nx = SeqOpts.resolution;
Ny = SeqOpts.resolution;
deltak=1/SeqOpts.FOV;
riseTime = 5e-20; % use the same rise times for all gradients, so we can neglect them
gx = mr.makeTrapezoid('x','FlatArea',Nx*deltak,'FlatTime',6.4e-3,'RiseTime', riseTime);
adc = mr.makeAdc(Nx,'Duration',gx.flatTime,'Delay',gx.riseTime);
gxPre = mr.makeTrapezoid('x','Area',-gx.area/2,'Duration',3e-3, 'RiseTime', riseTime);
phaseAreas = ((1:Ny)-Ny/2)*deltak;

% timing
delayTE=ceil((SeqOpts.TE - mr.calcDuration(gxPre) - mr.calcDuration(rf)/2 ...
    - mr.calcDuration(gx)/2)/seq.gradRasterTime)*seq.gradRasterTime;
delayTR=ceil((SeqOpts.TR - mr.calcDuration(gxPre) - mr.calcDuration(rf) ...
    - mr.calcDuration(gx) - delayTE)/seq.gradRasterTime)*seq.gradRasterTime;

phaseAreaIndex = length(phaseAreas) / 2;
jumpSign = 1;

% put blocks together
for i=1:Ny
    
    if strcmp(SeqOpts.Order, 'increase')
        phaseAreaIndex = i;
    end
    
    seq.addBlock(rf);
    gyPre = mr.makeTrapezoid('y','Area',phaseAreas(phaseAreaIndex),'Duration',3e-3,'RiseTime', riseTime);
    seq.addBlock(gxPre,gyPre);
    seq.addBlock(mr.makeDelay(delayTE));
    seq.addBlock(gx,adc);
    gyRew = mr.makeTrapezoid('y','Area',-phaseAreas(phaseAreaIndex),'Duration',3e-3,'RiseTime', riseTime);
    seq.addBlock(gyRew);
    seq.addBlock(mr.makeDelay(delayTR))
    
    if strcmp(SeqOpts.Order, 'center')
        phaseAreaIndex = phaseAreaIndex + jumpSign*i;
        jumpSign = -jumpSign;
    end
end

%write sequence
seq.write(seqFilename);

