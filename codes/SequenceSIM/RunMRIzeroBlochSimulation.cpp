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
	ReferenceVolume* refVol = new ReferenceVolume();
	BlochSimulator* bmSim = new BlochSimulator();

    //get the data from matlab and initialize the classes for simulation
	ReadMATLABInput(nrhs, prhs, refVol, bmSim);

	//Init kspace
	MatrixXcd KSpace;
	KSpace.resize(refVol->GetNumberOfColumns(), refVol->GetNumberOfRows());
	KSpace.fill(std::complex<double>(0, 0));

	//run the simulation
	bmSim->RunSimulation(KSpace);

	//set the pointers to retrieve the data in matlab
	ReturnKSpaceToMATLAB(nlhs, plhs, KSpace);

	return;
}