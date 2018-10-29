Tiny MRI sequence simulator.
Uses eigen library (LPGv3 license) for linear algebra operations.
Uses pulseq tool (MIT license)

compile mex files in matlab:

mex -I3rdParty\eigen-eigen-5a0156e40feb -I3rdParty\pulseq-master\src RunMRIzeroBlochSimulation.cpp BlochSimulator.cpp 3rdParty\pulseq-master\src\ExternalSequence.cpp

mex CXXOPTIMFLAGS="\$CXXOPTIMFLAGS -O3" -I3rdParty\eigen-eigen-5a0156e40feb -I3rdParty\pulseq-master\src\ RunMRIzeroBlochSimulation.cpp BlochSimulator.cpp 3rdParty\pulseq-master\src\ExternalSequence.cpp -output RunMRIzeroBlochSimulationNSpins

call with:

kspace = RunMRIzeroBlochSimulation(in1,in2)

3 Inputs:
1: Reference Volume MxNx3
(:,:,1) Proton Density
(:,:,2) T1 [s]
(:,:,3) T2 [s]

2: pulseq sequebce fileneme e.g. 'example.seq'

1 Output:
MxN complex k-space 
