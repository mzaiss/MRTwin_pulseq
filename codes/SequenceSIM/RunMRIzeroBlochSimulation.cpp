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
	std::vector<KSpaceEvent> kSpaceEvents;
	kSpaceEvents.resize(simulator.GetNumberOfKSpaceSamples());

	//run the simulation
	simulator.RunSimulation(sequence, kSpaceEvents);

	//set the pointers to retrieve the data in matlab
	ReturnKSpaceToMATLAB(nlhs, plhs, kSpaceEvents);

	return;
}