/*
The function that runs the simulation
MRIzero Project

kai.herz@tuebingen.mpg.de
*/


#include "MatlabIO.h"

// The function thats called from Matlab
void mexFunction(int nlhs, mxArray* plhs[],
	int nrhs, const mxArray* prhs[])
{
	//Init variables for simulation
	ReferenceVolume volume;
	BlochSimulator simulator;
	ExternalSequence sequence;        // sequence from pulseq

    //get the data from matlab and initialize the classes for simulation
	ReadMATLABInput(nrhs, prhs, &volume, &simulator, &sequence);

	//Init kspace
	MatrixXcd kSpace;
	kSpace.resize(volume.GetNumberOfRows(), volume.GetNumberOfColumns());
	kSpace.fill(std::complex<double>(0, 0));

	//run the simulation
	simulator.RunSimulation(sequence, kSpace);

	//set the pointers to retrieve the data in matlab
	ReturnKSpaceToMATLAB(nlhs, plhs, kSpace);

	return;
}