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
	ExternalSequence sequence;        // sequence from pulseq
	uint32_t numberOfSeqSamples = 0;
	uint32_t numberOfSpins = 256;

    //get the data from matlab and initialize the classes for simulation
	ReadMATLABInput(nrhs, prhs, &volume, &sequence, numberOfSeqSamples, numberOfSpins);

	//Init kspace
	KSpaceEvents kSpace = KSpaceEvents(numberOfSeqSamples);


	//case for Simulator template
	if(volume.GetNumberOfRows() == 32)
	{
		BlochSimulator simulator;
		if (!simulator.Initialize(&volume)) {
			mexErrMsgIdAndTxt("MRIzero:ReadMATLABInput:rrhs",
				"Could not initialize CUDA device");
		}
		simulator.RunSimulation(sequence, kSpace);
	}	
	else{
		mexErrMsgIdAndTxt("MRIzero:ReadMATLABInput:rrhs",
			"Not a valid image size");
	}


	//set the pointers to retrieve the data in matlab
	if (!ReturnKSpaceToMATLAB(nlhs, plhs, kSpace)) {
		mexErrMsgIdAndTxt("MRIzero:mexFunction:plhs",
			"Could not return k-space");
	}

	return;
}