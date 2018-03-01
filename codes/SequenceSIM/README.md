Tiny MRI sequence simulator.
Uses eigen library (LPGv3) for linear algebra operations.

compile mex files in matlab:

mex -I3rdParty\eigen-eigen-5a0156e40feb RunMRIzeroBlochSimulation.cpp BlochSimulator.cpp

call with:

kspace = RunMRIzeroBlochSimulation(in1,in2,in3)

3 Inputs:
1: Reference Volume MxNx3
(:,:,1) Proton Density
(:,:,2) T1 [s]
(:,:,3) T2 [s]

2: Px6 Vector with Pulse, Gradient and ADC events
(:,:,1) w1 magnitude (B1[uT] * Gamma[rad])
(:,:,2) w1 phase [rad]
(:,:,3) X gradient [uT/m]
(:,:,4) Y gradient [uT/m]
(:,:,5) Duration of the current event [s]
(:,:,6) ADC, 1 if sampling data after the event, 0 if not

3: Field stength [T]

1 Output:
MxN complex k-space 
