Tiny MRI sequence simulator.
Uses eigen library (LPGv3 license) for linear algebra operations.
Uses pulseq tool (MIT license)

compile mex files in matlab:

Windows:

  mex -I3rdParty\eigen-eigen-5a0156e40feb -I3rdParty\pulseq-master\src RunMRIzeroBlochSimulation.cpp BlochSimulator.cpp 3rdParty\pulseq-master\src\ExternalSequence.cpp

  mex CXXOPTIMFLAGS="\$CXXOPTIMFLAGS -O3" -I3rdParty\eigen-eigen-5a0156e40feb -I3rdParty\pulseq-master\src\ RunMRIzeroBlochSimulation.cpp BlochSimulator.cpp 3rdParty\pulseq-master\src\ExternalSequence.cpp -output RunMRIzeroBlochSimulationNSpins

Linux 64-bit (gcc-4.7):

  mex -I$MATLAB_PATH/extern/include/ -L$MATLAB_PATH/bin/glnxa64 -I3rdParty/eigen-eigen-5a0156e40feb/Eigen/src/ -I3rdParty/pulseq-master/src/ RunMRIzeroBlochSimulation.cpp BlochSimulator.cpp 3rdParty/pulseq-master/src/ExternalSequence.cpp

  mex CXXOPTIMFLAGS="\$CXXOPTIMFLAGS -O3" -I$MATLAB_PATH/extern/include/ -L$MATLAB_PATH/bin/glnxa64 -I3rdParty/eigen-eigen-5a0156e40feb/Eigen/src/ -I3rdParty/pulseq-master/src/ RunMRIzeroBlochSimulation.cpp BlochSimulator.cpp 3rdParty/pulseq-master/src/ExternalSequence.cpp -output RunMRIzeroBlochSimulationNSpins

  in all .cpp/.h files using Eigen lib change: 
  
  #include "Eigen\eigen"
  to:
  #include "Eigen/Core"
  #include <Eigen/Geometry>

  add #include <stdint.h> in BlochSimulator.h

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
